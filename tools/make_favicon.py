#!/usr/bin/env python3
"""
Generate the LECTA favicon: an open book with a sound wave on the right.
Palette:
  - Accent: #38bdf8 (sky blue)
  - Background: #0f172a (dark navy)
  - Secondary: #94a3b8 (slate gray)

Outputs:
  - libs/favicon/lecta.ico    (layers: 16, 32, 48, 64, 128, 256 px)
  - libs/favicon/lecta-512.png (single 512×512 PNG)

Run from the project root:
    python tools/make_favicon.py
"""

from pathlib import Path
from PIL import Image, ImageDraw

# ── Palette ───────────────────────────────────────────────────────────────────
BG     = (15, 23, 42)     # #0f172a
ACCENT = (56, 189, 248)   # #38bdf8
FADED  = (148, 163, 184)  # #94a3b8

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "libs" / "favicon"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_favicon(size: int) -> Image.Image:
    """Draw an open-book + sound-wave icon at *size*×*size*."""
    img = Image.new("RGBA", (size, size), BG)
    draw = ImageDraw.Draw(img)

    # ── Proportions (all relative to size) ──
    # Book occupies the left ~55 % of the canvas
    book_left   = size * 0.08
    book_right  = size * 0.52
    book_top    = size * 0.20
    book_bottom = size * 0.80
    book_mid_x  = (book_left + book_right) / 2

    # ── Book spine ──
    spine_width = size * 0.03
    draw.rectangle(
        [book_mid_x - spine_width / 2, book_top, book_mid_x + spine_width / 2, book_bottom],
        fill=ACCENT,
    )

    # ── Left page ──
    left_poly = [
        (book_left, book_bottom),          # bottom-left
        (book_left, book_top),             # top-left
        (book_mid_x, book_top),            # top (spine)
        (book_mid_x, book_bottom),         # bottom (spine)
    ]
    draw.polygon(left_poly, fill=ACCENT)

    # ── Right page (slightly open, showing a gap) ──
    right_top = book_top + size * 0.02
    right_bottom = book_bottom - size * 0.02
    right_poly = [
        (book_mid_x, book_bottom),          # bottom (spine)
        (book_mid_x + size * 0.02, right_top),  # top-right (slightly offset)
        (book_right, right_top + size * 0.02),  # right edge top
        (book_right, right_bottom),             # right edge bottom
    ]
    draw.polygon(right_poly, fill=FADED)

    # ── Sound-wave arcs on the right side ──
    wave_center_x = size * 0.68
    wave_center_y = size * 0.50
    wave_max_radius = size * 0.28

    # Three arc segments (growing outward), each ~120° wide
    for i, (r, color) in enumerate([
        (wave_max_radius * 0.35, ACCENT),
        (wave_max_radius * 0.60, ACCENT),
        (wave_max_radius * 0.85, ACCENT),
    ]):
        bbox = [
            wave_center_x - r,
            wave_center_y - r,
            wave_center_x + r,
            wave_center_y + r,
        ]
        draw.arc(bbox, start=-60, end=60, fill=color, width=max(int(size * 0.04), 1))

    # ── Small dot at the wave origin ──
    dot_r = max(int(size * 0.04), 1)
    draw.ellipse(
        [wave_center_x - dot_r, wave_center_y - dot_r,
         wave_center_x + dot_r, wave_center_y + dot_r],
        fill=ACCENT,
    )

    return img.convert("RGBA")


import io
import os
import struct
import sys

# Force UTF-8 for stdout on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def _build_ico(images, ico_path):
    """Build a multi-resolution ICO file manually.
    Pillow's ICO save doesn't reliably write all layers on all platforms,
    so we construct the ICO binary ourselves."""
    num_images = len(images)
    # ICO header: reserved(2) + type(2) + count(2)
    header = struct.pack("<HHH", 0, 1, num_images)
    # Directory entries: width, height, colors, reserved, planes, bpp, size, offset
    dir_entries = b""
    image_data = b""
    data_offset = 6 + num_images * 16  # header + directory entries

    for img in images:
        w, h = img.size
        # Convert to BGRA PNG bytes
        bgra = img.convert("RGBA")
        # Save as PNG bytes for the ICO
        buf = io.BytesIO()
        bgra.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        # ICO directory entry
        # width, height (0=256), colors, reserved, planes, bpp, size, offset
        ico_w = w if w < 256 else 0
        ico_h = h if h < 256 else 0
        entry = struct.pack(
            "<BBBBHHII",
            ico_w, ico_h, 0, 0, 1, 32,
            len(png_bytes), data_offset,
        )
        dir_entries += entry
        image_data += png_bytes
        data_offset += len(png_bytes)

    with open(ico_path, "wb") as f:
        f.write(header)
        f.write(dir_entries)
        f.write(image_data)


def main():
    # ── 512×512 PNG ──
    png_512 = draw_favicon(512)
    png_path = OUT_DIR / "lecta-512.png"
    png_512.save(png_path, "PNG")
    print(f"Saved {png_path}  (512x512)")

    # ── Multi-layer .ico via manual builder ──
    ico_sizes = [16, 32, 48, 64, 128, 256]
    images = [draw_favicon(s) for s in ico_sizes]
    ico_path = OUT_DIR / "lecta.ico"
    _build_ico(images, ico_path)
    print(f"Saved {ico_path}  (layers: {ico_sizes})")

    # ── Verification ──
    im = Image.open(ico_path)
    print(f"\nVerification:")
    print(f"  ICO file size:   {os.path.getsize(ico_path)} bytes")
    print(f"  ICO read size:   {im.size}")
    print(f"  ICO info sizes:  {sorted(im.info.get('sizes', []))}")
    print(f"  PNG file exists: {png_path.is_file()}")


if __name__ == "__main__":
    main()
