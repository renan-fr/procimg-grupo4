import cv2 as cv
import numpy as np
import matplotlib.pylab as plt

def isolar_cor(img_bgr, cor="vermelho", tolerancia_h=10, s_min=60, v_min=50):
    import numpy as np
    import cv2 as cv
    assert img_bgr is not None and img_bgr.ndim==3 and img_bgr.shape[2]==3, "img_bgr inválida"
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    def faixa(h, tol): 
        return (max(0, h - tol), min(179, h + tol))
    cores_h = {
        "vermelho": 0, "laranja": 15, "amarelo": 30, "verde": 60,
        "ciano": 90, "azul": 120, "roxo": 150, "magenta": 165, "rosa": 170,
    }
    c = (cor or "vermelho").strip().lower()
    if c == "vermelho":
        h1a,h1b = faixa(0, tolerancia_h)
        h2a,h2b = faixa(179, tolerancia_h)
        m1 = cv.inRange(hsv, (h1a, s_min, v_min), (h1b, 255, 255))
        m2 = cv.inRange(hsv, (h2a, s_min, v_min), (179, 255, 255))
        mask = cv.bitwise_or(m1, m2)
    else:
        h = cores_h.get(c, 120)
        ha,hb = faixa(h, tolerancia_h)
        mask = cv.inRange(hsv, (ha, s_min, v_min), (hb, 255, 255))
    isolada = cv.bitwise_and(img_bgr, img_bgr, mask=mask)
    return {
        "image": isolada,
        "images": [mask],
        "table": None,
        "plots": None,
        "meta": {"name":"isolamento_cor","params":{"cor":c,"tolerancia_h":tolerancia_h,"s_min":s_min,"v_min":v_min}}
    }
