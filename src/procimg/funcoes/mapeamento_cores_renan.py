import cv2 as cv
import numpy as np
import matplotlib.pylab as plt
# A função mapear_cores() converte a imagem em tons de cinza e aplica uma LUT (Look-Up Table) para gerar uma nova versão colorida, destacando padrões e contrastes. 
# São usadas paletas predefinidas do OpenCV (como VIRIDIS, HOT, JET e TURBO). 
# Após o processamento, a função exibe a comparação visual antes/depois e histogramas do canal de luminância (LAB-L) para análise quantitativa.

def mapear_cores(img_bgr, nome_lut="VIRIDIS"):
    """
    Aplica um mapeamento de cores (LUT) do OpenCV à imagem.
    Mostra comparação antes/depois e histogramas correspondentes.
    """

    # Dicionário resumido de LUTs disponíveis
    luts = {
        "VIRIDIS": cv.COLORMAP_VIRIDIS,
        "JET": cv.COLORMAP_JET,
        "HOT": cv.COLORMAP_HOT,
        "PLASMA": cv.COLORMAP_PLASMA,
        "INFERNO": cv.COLORMAP_INFERNO,
        "TURBO": cv.COLORMAP_TURBO
    }

    # Converte a imagem para tons de cinza e aplica a LUT escolhida
    img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    lut = luts.get(nome_lut.upper(), cv.COLORMAP_VIRIDIS)
    img_mapeada = cv.applyColorMap(img_gray, lut)

    # Exibe antes/depois
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original (Cinza)")
    plt.imshow(cv.cvtColor(img_gray, cv.COLOR_GRAY2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Mapeamento: {nome_lut.upper()}")
    plt.imshow(cv.cvtColor(img_mapeada, cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    # Gera histogramas antes/depois (canal L do LAB)
    def canal_l(img): return cv.split(cv.cvtColor(img, cv.COLOR_BGR2LAB))[0]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Antes — LAB-L")
    plt.hist(canal_l(img_bgr).ravel(), bins=256, range=(0, 255))
    plt.subplot(1, 2, 2)
    plt.title("Depois — LAB-L")
    plt.hist(canal_l(img_mapeada).ravel(), bins=256, range=(0, 255))
    plt.tight_layout()
    plt.show()

    return img_mapeada