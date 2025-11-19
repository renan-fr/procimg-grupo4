import cv2 as cv
import matplotlib.pylab as plt
import numpy as np

def equalizar_canais(img, canal_alvo):
    if img is None or not isinstance(img, np.ndarray):
        print("Erro: imagem inválida!")
        return

    img_equalizada = None

    if canal_alvo in ['B', 'G', 'R']:
        canais = list(cv.split(img))
        mapa_canais = {'B': 0, 'G': 1, 'R': 2}
        idx = mapa_canais[canal_alvo]
        canais[idx] = cv.equalizeHist(canais[idx])
        img_equalizada = cv.merge(canais)

    elif canal_alvo in ['H', 'S', 'V']:
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        canais = list(cv.split(hsv))
        mapa_canais = {'H': 0, 'S': 1, 'V': 2}
        idx = mapa_canais[canal_alvo]
        canais[idx] = cv.equalizeHist(canais[idx])
        img_equalizada = cv.cvtColor(cv.merge(canais), cv.COLOR_HSV2BGR)

    elif canal_alvo in ['L', 'A', 'B_lab']:
        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
        canais = list(cv.split(lab))
        mapa_canais = {'L': 0, 'A': 1, 'B_lab': 2}
        idx = mapa_canais[canal_alvo]
        canais[idx] = cv.equalizeHist(canais[idx])
        img_equalizada = cv.cvtColor(cv.merge(canais), cv.COLOR_LAB2BGR)

    else:
        print(f"Canal {canal_alvo} não suportado!")
        return

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(img_equalizada, cv.COLOR_BGR2RGB))
    plt.title(f"Canal {canal_alvo} Equalizado")
    plt.axis('off')

    plt.show()
