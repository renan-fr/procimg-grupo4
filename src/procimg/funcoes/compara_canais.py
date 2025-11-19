import cv2 as cv
import matplotlib.pylab as plt
import numpy as np

def comparar_canais(bgr, espaco="RGB", bins=None, figsize=(10, 9)):
    e = espaco.upper()

    if e == "RGB":
        img = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
        nomes = ["R","G","B"]
        faixas = [(0,256),(0,256),(0,256)]
    elif e == "HSV":
        img = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
        nomes = ["H","S","V"]
        faixas = [(0,180),(0,256),(0,256)]
    elif e == "LAB":
        img = cv.cvtColor(bgr, cv.COLOR_BGR2Lab)
        nomes = ["L","a","b"]
        faixas = [(0,256),(0,256),(0,256)]
    else:
        raise ValueError("Use: 'RGB', 'HSV' ou 'LAB'.")

    chs = cv.split(img)

    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    plt.figure(figsize=(6,4))
    plt.imshow(rgb)
    plt.title("Imagem Original (RGB)")
    plt.axis("off")
    plt.show()

    fig, axs = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(f"Espaço {e} — Canais e Histogramas", y=1.02, fontsize=14)

    for i, (ch, nome, (lo, hi)) in enumerate(zip(chs, nomes, faixas)):
        b = bins if isinstance(bins, int) else (180 if (e=="HSV" and nome=="H") else (hi-lo))

        # imagem do canal
        axs[i,0].imshow(ch, cmap="gray", vmin=lo, vmax=hi-1)
        axs[i,0].set_title(f"{e} — {nome} (imagem)")
        axs[i,0].axis("off")

        # histograma
        h, edges = np.histogram(ch.ravel(), bins=b, range=(lo, hi))
        centers = 0.5 * (edges[:-1] + edges[1:])
        axs[i,1].plot(centers, h)
        axs[i,1].set_title(f"{e} — {nome} (histograma)")
        axs[i,1].set_xlabel("Intensidade")
        axs[i,1].set_ylabel("Frequência")
        axs[i,1].grid(True, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.show()
