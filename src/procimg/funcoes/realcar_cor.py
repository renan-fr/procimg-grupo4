import cv2 as cv
import numpy as np

def realce_cor(img_bgr, ganho_s=1.2, ganho_v=1.0):
    """
    Ajusta saturação (S) e valor (V) no espaço HSV.
    - Entrada: img_bgr (uint8, BGR)
    - Saída: dict com "image" (BGR uint8) e meta
    """
    assert img_bgr is not None and img_bgr.ndim==3 and img_bgr.shape[2]==3, "img_bgr inválida"

    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv.split(hsv)

    s = np.clip(s * float(ganho_s), 0, 255)
    v = np.clip(v * float(ganho_v), 0, 255)

    hsv2 = cv.merge([h, s, v]).astype(np.uint8)
    out = cv.cvtColor(hsv2, cv.COLOR_HSV2BGR)

    return {
        "image": out,
        "images": None,
        "table": None,
        "plots": None,
        "meta": {"name":"realce_cor","params":{"ganho_s":ganho_s,"ganho_v":ganho_v}}
    }
