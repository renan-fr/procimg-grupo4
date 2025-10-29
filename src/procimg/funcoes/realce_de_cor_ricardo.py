# A função realce_cor() converte a imagem do modelo BGR para o modelo HSV, onde o canal S(Saturação) controla a intensidade da cor, então multiplicamos 
# o canal de saturação pelo valor de fator_saturacao e então a imagem é convertida de volta ao modelo BGR e exibe a imagem antes e depois do realce.

def realce_cor(img_path, fator_saturacao=1.5):

    img = cv.imread(img_path)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    h, s, v = cv.split(hsv)

    s = np.clip(s * fator_saturacao, 0, 255).astype(np.uint8)

    hsv_realce = cv.merge([h, s, v])

    img_realce = cv.cvtColor(hsv_realce, cv.COLOR_HSV2BGR)

    print("Imagem original:")
    cv2_imshow(img)
    print("Imagem com realce de cor:")
    cv2_imshow(img_realce)