import cv2 as cv
import numpy as np
import matplotlib.pylab as plt


_LUTS = {
    "AUTUMN": cv.COLORMAP_AUTUMN,
    "BONE": cv.COLORMAP_BONE,
    "JET": cv.COLORMAP_JET,
    "WINTER": cv.COLORMAP_WINTER,
    "RAINBOW": cv.COLORMAP_RAINBOW,
    "OCEAN": cv.COLORMAP_OCEAN,
    "SUMMER": cv.COLORMAP_SUMMER,
    "SPRING": cv.COLORMAP_SPRING,
    "COOL": cv.COLORMAP_COOL,
    "HSV": cv.COLORMAP_HSV,
    "PINK": cv.COLORMAP_PINK,
    "HOT": cv.COLORMAP_HOT,
    "PARULA": getattr(cv, "COLORMAP_PARULA", cv.COLORMAP_VIRIDIS),
    "MAGMA": cv.COLORMAP_MAGMA,
    "INFERNO": cv.COLORMAP_INFERNO,
    "PLASMA": cv.COLORMAP_PLASMA,
    "VIRIDIS": cv.COLORMAP_VIRIDIS,
    "CIVIDIS": cv.COLORMAP_CIVIDIS if hasattr(cv, "COLORMAP_CIVIDIS") else cv.COLORMAP_VIRIDIS,
    "TWILIGHT": cv.COLORMAP_TWILIGHT if hasattr(cv, "COLORMAP_TWILIGHT") else cv.COLORMAP_VIRIDIS,
    "TWILIGHT_SHIFTED": cv.COLORMAP_TWILIGHT_SHIFTED if hasattr(cv, "COLORMAP_TWILIGHT_SHIFTED") else cv.COLORMAP_VIRIDIS,
    "TURBO": cv.COLORMAP_TURBO if hasattr(cv, "COLORMAP_TURBO") else cv.COLORMAP_JET,
    "DEEPGREEN": cv.COLORMAP_DEEPGREEN if hasattr(cv, "COLORMAP_DEEPGREEN") else cv.COLORMAP_VIRIDIS,
}
_LUTS_LOWER = {k.lower(): v for k, v in _LUTS.items()}

def mapear_cores(img_bgr, nome_lut="VIRIDIS"):
    assert img_bgr is not None and img_bgr.ndim == 3 and img_bgr.shape[2] == 3, "img_bgr inválida"
    key = (nome_lut or "VIRIDIS").strip().lower()
    colormap = _LUTS_LOWER.get(key, _LUTS["VIRIDIS"])
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    mapped = cv.applyColorMap(gray, colormap)
    return {
        "image": mapped,
        "images": None,
        "table": None,
        "plots": None,
        "meta": {"name": "mapear_cores", "params": {"nome_lut": (nome_lut or "VIRIDIS")}}
    }
