import re

import numpy as np

HEX_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in [0, 1], shape (..., 3), to OKLab, shape (..., 3)."""
    lin = srgb_to_linear(np.clip(rgb, 0.0, 1.0))
    m1 = np.array(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ]
    )
    m2 = np.array(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ]
    )
    lms = lin @ m1.T
    return np.cbrt(lms) @ m2.T


def parse_hex(value: str) -> tuple[int, int, int] | None:
    m = HEX_RE.match(value.strip())
    if not m:
        return None
    h = m.group(1)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def hex_palette_to_oklab(colors: list[str]) -> np.ndarray | None:
    rgb = [parse_hex(c) for c in colors]
    if len(rgb) != 5 or any(c is None for c in rgb):
        return None
    return srgb_to_oklab(np.array(rgb, dtype=np.float64) / 255.0)


def rgb255_palette_to_oklab(palette: np.ndarray) -> np.ndarray:
    return srgb_to_oklab(np.asarray(palette, dtype=np.float64) / 255.0)


def image_tone(pixels: np.ndarray) -> list[float]:
    """Average OKLab color of an RGB uint8 image array (H, W, 3)."""
    lab = srgb_to_oklab(pixels.reshape(-1, 3).astype(np.float64) / 255.0)
    return [round(float(v), 4) for v in lab.mean(axis=0)]
