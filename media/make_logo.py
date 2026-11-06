#!/usr/bin/env python3
"""Generate the Orbitol logo as font-independent SVG, plus PNG renders.

Files written next to this script:

    orbitol_logo.svg, orbitol_logo_dark.svg    lockup: mark + wordmark
    orbitol_mark.svg, orbitol_mark_dark.svg    mark only, square (avatar, favicon)
    orbitol_logo.png (1600 px wide), orbitol_mark.png (512 px), and dark variants,
    when ImageMagick's `convert` with librsvg support is on PATH.

Design notes
------------
The logo is a sibling of the Crystalite logo: the same cool grey (#636370), the same
stroke weight, the same device of a mark rising from the `i` at 45 degrees in place of
its dot, and the same 45-degree cut on the top of the `t` beneath it. Where Crystalite
has a crystal, Orbitol has a Wyckoff orbit: an elliptical orbit path carrying three
symmetry-equivalent atoms, one of them sitting where the dot of the `i` would be.

The wordmark is set in URW Gothic Demi (URW base35, shipped with Ghostscript and most
Linux distributions) and converted to outlines with fontTools, so the SVG files render
identically everywhere and need no installed font. Lengths are in canvas units chosen
so that the letter height (ascender) is 249, as in the Crystalite logo PNG.

Dependencies: fontTools (`pip install fonttools`). PNG export needs ImageMagick.
"""

import math
import shutil
import subprocess
from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

HERE = Path(__file__).resolve().parent

FONT_CANDIDATES = [
    "/usr/share/fonts/urw-base35/URWGothic-Demi.otf",          # Fedora, RHEL
    "/usr/share/fonts/opentype/urw-base35/URWGothic-Demi.otf", # Debian, Ubuntu
    "/usr/share/fonts/URW/URWGothic-Demi.otf",
    "/usr/share/ghostscript/fonts/URWGothic-Demi.otf",
    "/opt/homebrew/share/ghostscript/fonts/URWGothic-Demi.otf",
]

WORD = "Orbitol"
ASCENDER = 249.0                  # canvas units; the font's ascender is 739 units
KERN = {"Or": -25, "rb": -8, "bi": -6, "it": -12, "to": -6, "ol": -8}  # font units

# The mark: an ellipse (semi-axes A, B) with its major axis at ANGLE degrees, drawn as a
# ring of width RING, carrying N_ATOMS atoms of radius ATOM equally spaced in the
# parametric angle. Its lower-left vertex sits above the `i`: the atom there is centred
# TIP_DX to the side of the stem and its bottom edge TIP_GAP above the stem top.
A, B, ANGLE = 112.0, 43.0, -45.0
RING, ATOM, N_ATOMS = 24.0, 26.0, 3
TIP_DX, TIP_GAP = -6.0, 22.0
T_GAP = 22.0                      # clearance between the ring and the cut top of the `t`
MARK_PAD = 0.06                   # square version: margin around the mark's enclosing circle

PALETTES = {
    "light": {"word": "#636370", "ring": "#636370", "atom": "#232328"},
    "dark": {"word": "#b4b4c0", "ring": "#b4b4c0", "atom": "#ececf1"},
}
PNG_SIZES = {"logo": 1600, "mark": 512}  # output width in pixels


def fmt(v):
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def find_font():
    for p in FONT_CANDIDATES:
        if Path(p).is_file():
            return p
    raise SystemExit("URW Gothic Demi not found; install the URW base35 fonts or edit FONT_CANDIDATES.")


# -- wordmark -----------------------------------------------------------------------

def layout(glyph_set, cmap):
    """Place the glyphs of WORD on a baseline at y = 0. Returns (name, x_units) pairs."""
    scale = ASCENDER / 739.0
    placed, x, prev = [], 0.0, None
    for ch in WORD:
        name = "dotlessi" if ch == "i" else cmap[ord(ch)]  # the mark replaces the dot
        if prev:
            x += KERN.get(prev + ch, 0)
        placed.append((ch, name, x * scale))
        x += glyph_set[name].width
        prev = ch
    return placed, scale


def glyph_path(glyph_set, name, x, scale, dx, dy):
    pen = SVGPathPen(glyph_set, ntos=fmt)
    glyph_set[name].draw(TransformPen(pen, (scale, 0, 0, -scale, x + dx, dy)))
    return pen.getCommands()


def glyph_polygon(glyph_set, name, x, scale):
    """Vertices of a straight-sided glyph (the `t`) in canvas coordinates."""
    rec = RecordingPen()
    glyph_set[name].draw(rec)
    pts = []
    for op, args in rec.value:
        if op in ("moveTo", "lineTo"):
            (ux, uy), = args
            pts.append((x + ux * scale, -uy * scale))
        elif op != "closePath":
            raise SystemExit("The `t` glyph has curves; the 45-degree cut expects straight sides.")
    return pts


def cut_polygon(pts, c):
    """Keep the part of a polygon with x + y >= c (one Sutherland-Hodgman clip edge)."""
    def inside(p):
        return p[0] + p[1] >= c

    def crossing(p, q):
        s = (c - p[0] - p[1]) / ((q[0] + q[1]) - (p[0] + p[1]))
        return (p[0] + s * (q[0] - p[0]), p[1] + s * (q[1] - p[1]))

    out = []
    for i, p in enumerate(pts):
        q = pts[(i + 1) % len(pts)]
        if inside(p):
            out.append(q if inside(q) else crossing(p, q))
        elif inside(q):
            out.extend([crossing(p, q), q])
    return out


def polygon_path(pts, dx, dy):
    return "M" + " L".join(f"{fmt(x + dx)} {fmt(y + dy)}" for x, y in pts) + " Z"


# -- mark ----------------------------------------------------------------------------

def ellipse_point(cx, cy, t_deg):
    t, r = math.radians(t_deg), math.radians(ANGLE)
    px, py = A * math.cos(t), B * math.sin(t)
    return cx + px * math.cos(r) - py * math.sin(r), cy + px * math.sin(r) + py * math.cos(r)


def atom_angles():
    return [180.0 + 360.0 * k / N_ATOMS for k in range(N_ATOMS)]  # one atom at the lower-left vertex


def mark_elements(cx, cy, pal):
    els = [f'<ellipse cx="{fmt(cx)}" cy="{fmt(cy)}" rx="{fmt(A)}" ry="{fmt(B)}" '
           f'transform="rotate({fmt(ANGLE)} {fmt(cx)} {fmt(cy)})" '
           f'fill="none" stroke="{pal["ring"]}" stroke-width="{fmt(RING)}"/>']
    for t in atom_angles():
        px, py = ellipse_point(cx, cy, t)
        els.append(f'<circle cx="{fmt(px)}" cy="{fmt(py)}" r="{fmt(ATOM)}" fill="{pal["atom"]}"/>')
    return els


def mark_bounds(cx, cy):
    r = math.radians(ANGLE)
    hw = math.hypot(A * math.cos(r), B * math.sin(r)) + RING / 2
    hh = math.hypot(A * math.sin(r), B * math.cos(r)) + RING / 2
    x0, y0, x1, y1 = cx - hw, cy - hh, cx + hw, cy + hh
    for t in atom_angles():
        px, py = ellipse_point(cx, cy, t)
        x0, y0, x1, y1 = min(x0, px - ATOM), min(y0, py - ATOM), max(x1, px + ATOM), max(y1, py + ATOM)
    return x0, y0, x1, y1


# -- assembly ------------------------------------------------------------------------

def svg_document(width, height, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{fmt(width)}" height="{fmt(height)}" '
            f'viewBox="0 0 {fmt(width)} {fmt(height)}">\n<title>Orbitol</title>\n'
            + "\n".join(body) + "\n</svg>\n")


def lockup_svg(font, pal):
    glyph_set, cmap = font.getGlyphSet(), font.getBestCmap()
    placed, scale = layout(glyph_set, cmap)

    # The mark's lower-left vertex (an atom) sits above the `i`; the centre lies A along the axis.
    x_i = next(x for ch, _, x in placed if ch == "i")
    i_centre = x_i + 120.5 * scale                     # stem of dotlessi spans 54..187 units
    i_top = -554.0 * scale                             # x-height
    tip = (i_centre + TIP_DX, i_top - TIP_GAP - ATOM)
    cx = tip[0] + A * math.cos(math.radians(ANGLE))
    cy = tip[1] + A * math.sin(math.radians(ANGLE))
    # Cut line for the `t`: parallel to the axis, T_GAP beyond the ring's lower-right edge.
    cut_c = tip[0] + tip[1] + (B + RING / 2 + T_GAP) * math.sqrt(2)

    # Canvas = ink bounds of the word and the mark.
    x0, y0, x1, y1 = mark_bounds(cx, cy)
    for ch, name, x in placed:
        bp = BoundsPen(glyph_set)
        glyph_set[name].draw(bp)
        bx0, by0, bx1, by1 = bp.bounds
        x0, x1 = min(x0, x + bx0 * scale), max(x1, x + bx1 * scale)
        y0, y1 = min(y0, -by1 * scale), max(y1, -by0 * scale)
    # Integer canvas (rasterisers round the intrinsic size), sub-pixel slack split evenly.
    width, height = math.ceil(x1 - x0), math.ceil(y1 - y0)
    dx, dy = -x0 + (width - (x1 - x0)) / 2, -y0 + (height - (y1 - y0)) / 2

    body = [f'<g fill="{pal["word"]}">']
    for ch, name, x in placed:
        if ch == "t":
            body.append(f'<path d="{polygon_path(cut_polygon(glyph_polygon(glyph_set, name, x, scale), cut_c), dx, dy)}"/>')
        else:
            body.append(f'<path d="{glyph_path(glyph_set, name, x, scale, dx, dy)}"/>')
    body.append("</g>")
    body += mark_elements(cx + dx, cy + dy, pal)
    return svg_document(width, height, body)


def mark_svg(pal):
    """Square mark. Its smallest enclosing circle is centred, so a circular avatar crop keeps it whole."""
    far = A + ATOM                   # the atom on the lower-left vertex
    near = A + RING / 2              # the bare ring at the upper-right vertex
    side = math.ceil((far + near) * (1 + MARK_PAD))
    shift = (far - near) / 2         # ellipse centre moves up-right along the axis
    r = math.radians(ANGLE)
    cx, cy = side / 2 + shift * math.cos(r), side / 2 + shift * math.sin(r)
    return svg_document(side, side, mark_elements(cx, cy, pal))


def render_png(svg_path, png_path, width_px):
    svg_width = float(svg_path.read_text().split('width="', 1)[1].split('"', 1)[0])
    density = 96.0 * width_px / svg_width * (1 + 1e-6)  # librsvg truncates; stay above the integer
    subprocess.run(["convert", "-background", "none", "-density", f"{density:.6f}",
                    str(svg_path), str(png_path)], check=True)


def main():
    font = TTFont(find_font())
    outputs = []
    for variant, pal in PALETTES.items():
        suffix = "" if variant == "light" else "_dark"
        for kind, svg in (("logo", lockup_svg(font, pal)), ("mark", mark_svg(pal))):
            path = HERE / f"orbitol_{kind}{suffix}.svg"
            path.write_text(svg)
            outputs.append((kind, path))
    if shutil.which("convert"):
        for kind, path in outputs:
            render_png(path, path.with_suffix(".png"), PNG_SIZES[kind])
    else:
        print("ImageMagick `convert` not found; wrote SVG files only.")
    for _, path in outputs:
        print("wrote", path.name)


if __name__ == "__main__":
    main()
