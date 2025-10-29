import cv2 as cv
import numpy as np
import matplotlib.pylab as plt
def substituir_cor(caminho_imagem, cor_alvo, nova_cor):
    """
    Substitui uma das cores principais (vermelho, verde ou azul) por outra cor.

    Parâmetros:
        caminho_imagem: str -> Caminho da imagem (ex: 'foto.jpg')
        cor_alvo: str -> Cor a ser substituída: 'vermelho', 'verde' ou 'azul'
        nova_cor: tuple -> Nova cor em BGR (ex: (0, 0, 255) para vermelho)
    """

    # Dicionário de faixas HSV das cores principais
    cores_hsv = {
        'vermelho': [(np.array([0, 120, 70]), np.array([10, 255, 255])),
                     (np.array([170, 120, 70]), np.array([180, 255, 255]))],  # vermelho tem duas faixas no HSV
        'verde': [(np.array([36, 100, 100]), np.array([86, 255, 255]))],
        'azul': [(np.array([94, 80, 2]), np.array([126, 255, 255]))]
    }

    # Carrega a imagem
    img = cv.imread(caminho_imagem)
    if img is None:
        print("Erro: imagem não encontrada!")
        return

    # Converte para HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Verifica se a cor alvo é válida
    if cor_alvo not in cores_hsv:
        print("Erro: cor inválida. Use 'vermelho', 'verde' ou 'azul'.")
        return

    # Cria a máscara para a cor alvo (considerando múltiplas faixas, ex: vermelho)
    mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for (lower, upper) in cores_hsv[cor_alvo]:
        mask = cv.inRange(hsv, lower, upper)
        mask_total = cv.bitwise_or(mask_total, mask)

    # Cria cópia da imagem e aplica a substituição
    resultado = img.copy()
    resultado[mask_total > 0] = nova_cor


    # Exibe lado a lado usando matplotlib
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"{cor_alvo.capitalize()} Substituído")
    plt.imshow(cv.cvtColor(resultado, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()