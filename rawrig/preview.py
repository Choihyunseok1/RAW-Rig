from __future__ import annotations

import cv2
import numpy as np


def preview_rgb(frame, red=1., blue=1., brightness_ev=0., max_width=1920):
    """Rendering-only conversion. Never modify the array that is stored as RAW."""
    patterns = {"RG": "RGGB", "GR": "GRBG", "GB": "GBRG", "BG": "BGGR"}
    pattern = patterns[frame.pixel_format[5:7]]
    rgb = cv2.cvtColor(frame.pixels, getattr(cv2, f"COLOR_Bayer{pattern}2RGB")).astype(np.float32)
    maximum = max(1, frame.settings.get("pixel_max", np.iinfo(frame.pixels.dtype).max))
    rgb /= maximum
    rgb[..., 0] *= red
    rgb[..., 2] *= blue
    rgb *= 2 ** brightness_ev
    rgb = np.power(np.clip(rgb, 0, 1), 1 / 2.2)
    if rgb.shape[1] > max_width:
        rgb = cv2.resize(rgb, (max_width, round(rgb.shape[0] * max_width / rgb.shape[1])), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb * 255, dtype=np.uint8)
