/* Compiled port of NMDANeuron.step (model.py, lif.py) and calcium_step
 * (calcium.py), plasticity off. Same arithmetic in the same order;
 * tests/test_kernel.py holds it to the Python model. Inference stays in Python
 * (fast.PlasticSim).
 *
 * u_phi is the step mean of u^m / u(0)^m (exact quadrature); u_phi = 1 is the
 * historical frozen step.
 *
 * pooled = 1 is NMDANeuron(pooled_gate=True): w holds event counts, AMPA adds
 * gbar * wscale * count, u holds v = sum of u^m over independent clefts and s
 * the summed occupancy, saturating in mean field through (1 - s/n_syn).
 * Jmax is then J_max / n_syn. gI is the optional somatic inhibition; with
 * w_inh NULL it stays exactly 0.
 */
#include <math.h>

#define MAXN 16

typedef struct {
    long n, nsub;
    double dt, Cd, Cs, gL, EL, gc, Ea, decay, Vth, Vreset, tref;
    double dec_u, a_u, k_b, tau_N, dec_bf, dec_bs, A_f, A_s, mgc;
    double c0, tau_c, kon, koff, u_phi;
    double Jmax[MAXN], Btot[MAXN];
    long pooled;
    double dec_v, k_eff, a_um, EI, dec_I, gbar_I;
    double wscale[MAXN], n_syn[MAXN];
} Params;

typedef struct {
    double g[MAXN], V[MAXN], u[MAXN], s[MAXN], T[MAXN], b[MAXN], gbar[MAXN];
    double Vs, t, refr, yf, ys;
    long steps;
    double gI;
} State;

static void calcium_step(double *total_io, double *bound_io, double influx, double capacity,
                         const Params *p)
{
    double total = *total_io, bound = *bound_io;
    const double h = p->dt / p->nsub, kon = p->kon, koff = p->koff;
    const double decay = exp(-0.5 * h / p->tau_c);
    for (long i = 0; i < p->nsub; i++) {
        double equilibrium = bound + p->c0 + influx * p->tau_c;
        total = equilibrium + (total - equilibrium) * decay;
        double beta = kon * (total + capacity) + koff;
        double gamma = kon * total * capacity;
        double diff = kon * (total - capacity);
        double speed = sqrt(diff * diff + 2 * kon * koff * (total + capacity) + koff * koff);
        double lower = 2 * gamma / (beta + speed);
        double delta = speed / kon;
        double denominator = bound - lower - delta;
        double ratio = denominator != 0.0 ? ((bound - lower) / denominator) * exp(-speed * h) : 0.0;
        bound = lower - delta * ratio / (1 - ratio);
        equilibrium = bound + p->c0 + influx * p->tau_c;
        total = equilibrium + (total - equilibrium) * decay;
    }
    *total_io = total;
    *bound_io = bound;
}

/* w: n_steps x n weights. samples: free calcium, one row per sample_stride
 * absolute steps. spike_times receives at most max_spikes values.
 * Returns the number of spikes, including any beyond max_spikes. */
long run_chunk(const Params *p, State *x, const double *w, long n_steps, long sample_stride,
               double *samples, long *n_samples, double *spike_times, long max_spikes,
               const double *w_inh)
{
    const long n = p->n;
    const double dt = p->dt;
    long spikes = 0, written = 0;
    for (long k = 0; k < n_steps; k++, w += n) {
        double sum_coupling = 0.0;
        for (long i = 0; i < n; i++) {
            x->g[i] = x->g[i] * p->decay + x->gbar[i] * p->wscale[i] * w[i];
            sum_coupling += p->gc * (x->V[i] - x->Vs);
        }
        x->gI *= p->dec_I;
        if (w_inh)
            x->gI += p->gbar_I * w_inh[k];
        const double I_soma = p->gL * (p->EL - x->Vs) + sum_coupling + x->gI * (p->EI - x->Vs);
        const double dVs = dt * I_soma / p->Cs;
        for (long i = 0; i < n; i++) {
            double I_d = p->gL * (p->EL - x->V[i]) + x->g[i] * (p->Ea - x->V[i])
                       + p->gc * (x->Vs - x->V[i]);
            x->V[i] += dt * I_d / p->Cd;
        }
        const int refractory = x->t < x->refr;
        if (!refractory)
            x->Vs += dVs;
        x->t += dt;
        int spiked = 0;
        if (!refractory && x->Vs >= p->Vth) {
            spiked = 1;
            x->Vs = p->Vreset;
            x->refr = x->t + p->tref;
            if (spikes < max_spikes)
                spike_times[spikes] = x->t;
            spikes++;
        }
        x->yf *= p->dec_bf;
        x->ys *= p->dec_bs;
        if (spiked) {
            x->yf = p->A_f;
            x->ys = p->A_s;
        }
        for (long i = 0; i < n; i++) {
            if (p->pooled) {
                x->u[i] *= p->dec_v;
                if (w[i] > 0)
                    x->u[i] += w[i] * p->a_um;
                double A = p->k_eff * x->u[i] * p->u_phi;
                double rate = A / p->n_syn[i] + 1.0 / p->tau_N;
                double s_inf = A / rate;
                x->s[i] = s_inf + (x->s[i] - s_inf) * exp(-dt * rate);
            } else {
                x->u[i] *= p->dec_u;
                if (w[i] > 0)
                    x->u[i] += p->a_u * (1.0 - x->u[i]);
                double A = p->k_b * x->u[i] * x->u[i] * p->u_phi;
                double rate = A + 1.0 / p->tau_N;
                double s_inf = A / rate;
                x->s[i] = s_inf + (x->s[i] - s_inf) * exp(-dt * rate);
            }
            double V_eff = x->V[i] + x->yf + x->ys;
            double J = p->Jmax[i] * x->s[i] / (1.0 + p->mgc * exp(-0.062 * V_eff));
            calcium_step(&x->T[i], &x->b[i], J, p->Btot[i], p);
        }
        x->steps++;
        if (x->steps % sample_stride == 0) {
            for (long i = 0; i < n; i++)
                samples[written * n + i] = x->T[i] - x->b[i];
            written++;
        }
    }
    *n_samples = written;
    return spikes;
}
