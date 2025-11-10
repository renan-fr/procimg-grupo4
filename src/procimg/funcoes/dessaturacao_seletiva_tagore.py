import cv2 as cv
import numpy as np

def dessaturacao_seletiva(img_bgr, cor="vermelho", fator=0.5):
    """
    Aplica dessaturação seletiva automática:
      - Mantém colorido apenas pixels próximos à cor especificada.
      - O restante é convertido para tons de cinza.
    Parâmetros:
      img_bgr: imagem BGR (np.ndarray)
      cor: nome da cor principal a preservar (ex.: 'vermelho', 'verde', 'azul')
      fator: intensidade da dessaturação (0 = sem efeito, 1 = PB total)
    Retorna:
      dict com {'image': imagem_resultado}
    """

    if img_bgr is None or not isinstance(img_bgr, np.ndarray):
        raise ValueError("Imagem inválida ou não carregada.")

    # Mapa aproximado de cores em HSV (Hue central e faixa ±20)
    cores_hue = {
        "vermelho": 0, "laranja": 15, "amarelo": 30, "verde": 60,
        "ciano": 90, "azul": 120, "roxo": 150, "magenta": 165, "rosa": 170
    }
    hue_central = cores_hue.get(cor.lower(), 0)
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)

    # Máscara para a cor alvo
    lower = np.array([max(0, hue_central - 20), 50, 50])
    upper = np.array([min(179, hue_central + 20), 255, 255])
    mask = cv.inRange(hsv, lower, upper)

    # Converte imagem para PB
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

    # Combina mantendo cor no alvo e PB no resto
    alpha = np.clip(fator, 0.0, 1.0)
    result = np.where(mask[:, :, None] > 0, img_bgr, (1 - alpha) * img_bgr + alpha * gray_bgr).astype(np.uint8)

    return {"image": result}
