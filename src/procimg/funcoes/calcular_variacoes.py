import cv2 as cv
import numpy as np

def calcular_variacoes_media(img_original, img_modificada):
    # Converter ambas para HSV
    hsv_original = cv.cvtColor(img_original, cv.COLOR_BGR2HSV)
    hsv_modificada = cv.cvtColor(img_modificada, cv.COLOR_BGR2HSV)

    # Calcular diferenças absolutas canal a canal
    diff_h = np.mean(np.abs(hsv_modificada[:,:,0].astype(np.float32) - hsv_original[:,:,0].astype(np.float32)))
    diff_s = np.mean(np.abs(hsv_modificada[:,:,1].astype(np.float32) - hsv_original[:,:,1].astype(np.float32)))
    diff_v = np.mean(np.abs(hsv_modificada[:,:,2].astype(np.float32) - hsv_original[:,:,2].astype(np.float32)))

    # Exibir resultados
    print("📊 Variações médias entre as imagens:")
    print(f"→ Hue (matiz): {diff_h:.2f}")
    print(f"→ Saturação: {diff_s:.2f}")
    print(f"→ Valor (brilho): {diff_v:.2f}")

    return {"Hue": diff_h, "Saturação": diff_s, "Valor": diff_v}
