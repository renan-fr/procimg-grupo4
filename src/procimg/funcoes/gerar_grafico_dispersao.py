import cv2 as cv
import matplotlib.pyplot as plt

def gerar_grafico_dispersao(caminho_imagem):
    img = cv.imread(caminho_imagem)

    if img is None:
        print("Erro: imagem não encontrada.")
        return

    print("Imagem em Análise:")
    cv.imshow("Imagem em Análise", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    print("")

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    intensidade = v.flatten()
    saturacao = s.flatten()

    plt.figure(figsize=(6,5))
    plt.scatter(intensidade, saturacao, s=1)
    plt.title("Dispersão entre Intensidade (V) e Saturação (S)")
    plt.xlabel("Intensidade (V)")
    plt.ylabel("Saturação (S)")
    plt.grid(True)
    plt.show()

    print("")

    b, g, r = cv.split(img)
    canais = [('Canal R', r), ('Canal G', g), ('Canal B', b)]

    plt.figure(figsize=(15,4))
    for i, (nome, canal) in enumerate(canais):
        plt.subplot(1, 3, i+1)
        plt.scatter(canal.flatten(), saturacao, s=1)
        plt.title(f"{nome} x Saturação (S)")
        plt.xlabel("Intensidade no canal")
        plt.ylabel("Saturação (S)")
        plt.grid(True)

    plt.tight_layout()
    plt.show()
