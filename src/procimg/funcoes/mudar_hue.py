import cv2 as cv
import numpy as np
import matplotlib.pylab as plt

def mudar_hue(img_bgr, deslocamento_hue):
    import numpy as np
    import cv2 as cv
    assert img_bgr is not None and img_bgr.ndim==3 and img_bgr.shape[2]==3, "img_bgr inválida"
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    h = (h.astype(np.int16) + int(deslocamento_hue)) % 180
    hsv2 = cv.merge([h.astype(np.uint8), s, v])
    out = cv.cvtColor(hsv2, cv.COLOR_HSV2BGR)
    return {
        "image": out,
        "images": None,
        "table": None,
        "plots": None,
        "meta": {"name":"mudar_hue","params":{"deslocamento_hue":int(deslocamento_hue)}}
    }
