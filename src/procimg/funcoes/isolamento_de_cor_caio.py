import cv2 as cv
import numpy as np
import matplotlib.pylab as plt
def isolar_cor(imagem_path, cor='vermelho'):
    # Lê a imagem
    imagem = cv.imread(imagem_path)
    imagem_rgb = cv.cvtColor(imagem, cv.COLOR_BGR2RGB)

    # Converte para HSV (melhor para segmentar cores)
    hsv = cv.cvtColor(imagem, cv.COLOR_BGR2HSV)

    # Define faixas de cor (em HSV)
    if cor == 'vermelho':
        # Vermelho é dividido em duas faixas no espectro HSV
        faixa1 = np.array([0, 120, 70])
        faixa2 = np.array([10, 255, 255])
        faixa3 = np.array([170, 120, 70])
        faixa4 = np.array([180, 255, 255])

        mask1 = cv.inRange(hsv, faixa1, faixa2)
        mask2 = cv.inRange(hsv, faixa3, faixa4)
        mascara = mask1 + mask2

    elif cor == 'verde':
        faixa_baixa = np.array([35, 50, 50])
        faixa_alta = np.array([85, 255, 255])
        mascara = cv.inRange(hsv, faixa_baixa, faixa_alta)

    elif cor == 'azul':
        faixa_baixa = np.array([90, 50, 50])
        faixa_alta = np.array([130, 255, 255])
        mascara = cv.inRange(hsv, faixa_baixa, faixa_alta)

    else:
        raise ValueError("Cor inválida! Escolha entre: 'vermelho', 'verde' ou 'azul'.")

    # Aplica a máscara na imagem original
    resultado = cv.bitwise_and(imagem_rgb, imagem_rgb, mask=mascara)

    # Exibe imagens
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(imagem_rgb)
    plt.title("Imagem Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(resultado)
    plt.title(f"Cor isolada: {cor}")
    plt.axis("off")

    plt.show()
