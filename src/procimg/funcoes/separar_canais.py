import cv2 as cv
import matplotlib.pylab as plt

def separar_canais(imagem_path):
    img_bgr = cv.imread(imagem_path)
    if img_bgr is None:
        print("Erro: imagem não encontrada!")
        return

    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    img_lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)

    (R, G, B) = cv.split(img_rgb)

    (H, S, V) = cv.split(img_hsv)

    (L, A, B_lab) = cv.split(img_lab)

    plt.figure(figsize=(13, 12))

    plt.subplot(3, 3, 1), plt.imshow(R, cmap='gray'), plt.title('Canal Vermelho (R)')
    plt.subplot(3, 3, 2), plt.imshow(G, cmap='gray'), plt.title('Canal Verde (G)')
    plt.subplot(3, 3, 3), plt.imshow(B, cmap='gray'), plt.title('Canal Azul (B)')

    plt.subplot(3, 3, 4), plt.imshow(H, cmap='gray'), plt.title('Canal Matiz (H)')
    plt.subplot(3, 3, 5), plt.imshow(S, cmap='gray'), plt.title('Canal Saturação (S)')
    plt.subplot(3, 3, 6), plt.imshow(V, cmap='gray'), plt.title('Canal Valor (V)')

    plt.subplot(3, 3, 7), plt.imshow(L, cmap='gray'), plt.title('Canal Luminosidade (L)')
    plt.subplot(3, 3, 8), plt.imshow(A, cmap='gray'), plt.title('Canal a')
    plt.subplot(3, 3, 9), plt.imshow(B_lab, cmap='gray'), plt.title('Canal b')

    plt.show()
