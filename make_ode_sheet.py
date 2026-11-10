"""Render MODEL_ODES.md into the complete, switch-free ODE_sheet.pdf.

The Markdown is the single source for prose and equations. Explicit pagebreak
comments define pages; matplotlib mathtext avoids a system TeX dependency.
"""
from pathlib import Path
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "DejaVu Serif"})
PAGE = (8.27, 11.69)


def blocks(page):
    for paragraph in re.split(r"\n\s*\n", page.strip()):
        text = " ".join(paragraph.splitlines())
        if text.startswith("$$"):
            yield text[1:-1], 13, "normal", 10, True
        elif text.startswith("# "):
            yield text[2:], 18, "bold", 15, False
        elif text.startswith("## "):
            yield text[3:], 12.5, "bold", 10, False
        else:
            yield textwrap.fill(text, width=94), 10, "normal", 9, False


def render(output=ROOT / "ODE_sheet.pdf"):
    source = (ROOT / "MODEL_ODES.md").read_text()
    date = re.search(r"equation sheet · (.+?)\.", source).group(1)
    pages = source.split("<!-- pagebreak -->")
    with PdfPages(output, metadata={"Title": "Full neuron system: ODEs and Bayesian plasticity",
                                   "Author": "Neuron model project"}) as pdf:
        for number, page in enumerate(pages, 1):
            fig = plt.figure(figsize=PAGE, dpi=110)
            artists = []
            for content, size, weight, gap, equation in blocks(page):
                artist = fig.text(.085, .94, content, fontsize=size,
                                  fontweight=weight, va="top", linespacing=1.35,
                                  color="#172b3a" if weight == "bold" else "#202020")
                artists.append((artist, size, gap, equation))
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            # Fit long equations and prose to the printable width.
            for artist, size, gap, equation in artists:
                width = artist.get_window_extent(renderer).width / fig.bbox.width
                if width > .81:
                    artist.set_fontsize(size * .81 / width)
            fig.canvas.draw()
            heights = [a.get_window_extent(renderer).height / fig.bbox.height
                       for a, _, _, _ in artists]
            gaps = [gap / (PAGE[1] * 72) for _, _, gap, _ in artists]
            scale = min(1., .86 / sum(h + g for h, g in zip(heights, gaps)))
            y = .945
            for (artist, _, _, _), height, gap in zip(artists, heights, gaps):
                artist.set_position((.085, y))
                artist.set_fontsize(artist.get_fontsize() * scale)
                y -= (height + gap) * scale
            fig.text(.085, .038, f"NEURON MODEL  /  {date.upper()}", fontsize=8, color=".45")
            fig.text(.915, .038, f"{number} / {len(pages)}", fontsize=8,
                     color=".45", ha="right")
            fig.canvas.draw()
            previous_bottom = None
            for artist, _, _, _ in artists:
                box = artist.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
                if box.x1 > .92 or box.y0 < .065:
                    raise RuntimeError(f"Content outside page {number} ({box.bounds}): {artist.get_text()[:60]}")
                if previous_bottom is not None and box.y1 > previous_bottom + .003:
                    raise RuntimeError(f"Overlapping content on page {number}")
                previous_bottom = box.y0
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Wrote {output} ({len(pages)} pages); layout checks passed")


if __name__ == "__main__":
    render()
