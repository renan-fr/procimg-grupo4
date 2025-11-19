import cv2 as cv
import numpy as np

def calcular_estatisticas(imagem_path):

    img_bgr = cv.imread(imagem_path)
    if img_bgr is None:
        print("Erro: não foi possível carregar a imagem.")
        return

    resultados_gerais = {}

    modelos = {
        'BGR': (img_bgr, ['Azul (B)', 'Verde (G)', 'Vermelho (R)']),
        'HSV': (cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV), ['Matiz (H)', 'Saturação (S)', 'Valor (V)']),
        'Lab': (cv.cvtColor(img_bgr, cv.COLOR_BGR2Lab), ['L* (Luminosidade)', 'a* (Verde–Vermelho)', 'b* (Azul–Amarelo)'])
    }

    print("Imagem Original:")
    cv.imshow("Imagem Original", img_bgr)
    cv.waitKey(0)
    cv.destroyAllWindows()

    for modelo, (img_convertida, nomes_canais) in modelos.items():
        canais = cv.split(img_convertida)
        resultados = {}

        for i, canal in enumerate(canais):
            canal = canal.astype(np.float64)
            media = np.mean(canal)
            desvio = np.std(canal)
            skewness = np.mean(((canal - media) / desvio) ** 3)

            resultados[nomes_canais[i]] = {
                'Média': float(media),
                'Desvio Padrão': float(desvio),
                'Assimetria (Skewness)': float(skewness)
            }

        resultados_gerais[modelo] = resultados

    for modelo, valores in resultados_gerais.items():
        print(f"\n==============================")
        print(f" Modelo de Cor: {modelo}")
        print(f"==============================")
        for canal, stats in valores.items():
            print(f"\n--- Canal {canal} ---")
            print(f"Média: {stats['Média']:.2f}")
            print(f"Desvio Padrão: {stats['Desvio Padrão']:.2f}")
            print(f"Assimetria (Skewness): {stats['Assimetria (Skewness)']:.4f}")
