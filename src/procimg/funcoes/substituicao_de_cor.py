import cv2 as cv
import numpy as np
import matplotlib.pylab as plt

def trocar_cor(img_bgr, cor_original='vermelho', cor_nova='azul', tolerancia_h=10, s_min=60, v_min=50, alpha=1.0):
    import numpy as np
    import cv2 as cv
    assert img_bgr is not None and img_bgr.ndim==3 and img_bgr.shape[2]==3, "img_bgr inválida"
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV).astype(np.float32)

    def faixa(h, tol): 
        return (max(0, h - tol), min(179, h + tol))

    cores_h = {
        "vermelho": 0, "laranja": 15, "amarelo": 30, "verde": 60,
        "ciano": 90, "azul": 120, "roxo": 150, "magenta": 165, "rosa": 170,
    }

    c0 = (cor_original or 'vermelho').strip().lower()

    if c0 == "vermelho":
        h1a,h1b = faixa(0, tolerancia_h)
        h2a,h2b = faixa(179, tolerancia_h)
        m1 = cv.inRange(hsv.astype(np.uint8), (h1a, s_min, v_min), (h1b, 255, 255))
        m2 = cv.inRange(hsv.astype(np.uint8), (h2a, s_min, v_min), (179, 255, 255))
        mask = cv.bitwise_or(m1, m2)
    else:
        h0 = cores_h.get(c0, 120)
        ha,hb = faixa(h0, tolerancia_h)
        mask = cv.inRange(hsv.astype(np.uint8), (ha, s_min, v_min), (hb, 255, 255))

    mask_f = cv.GaussianBlur(mask, (7,7), 0).astype(np.float32)/255.0

    alvo_h = {
        "vermelho": 0, "laranja": 15, "amarelo": 30, "verde": 60,
        "ciano": 90, "azul": 120, "roxo": 150, "magenta": 165, "rosa": 170,
    }.get((cor_nova or 'azul').strip().lower(), 120)

    h,s,v = cv.split(hsv)
    h_alvo = np.full_like(h, float(alvo_h))
    h_out = (1-mask_f)*h + mask_f*(alpha*h_alvo + (1-alpha)*h)

    hsv2 = cv.merge([np.mod(h_out,180), s, v]).astype(np.uint8)
    out = cv.cvtColor(hsv2, cv.COLOR_HSV2BGR)

    return {
        "image": out,
        "images": [mask],
        "table": None,
        "plots": None,
        "meta": {"name":"trocar_cor","params":{"cor_original":c0,"cor_nova":cor_nova,
                                                "tolerancia_h":tolerancia_h,"s_min":s_min,"v_min":v_min,"alpha":alpha}}
    }
