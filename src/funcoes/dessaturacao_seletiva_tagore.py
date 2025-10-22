# Para chamar a função é preciso informar caminho da imagem original, caminho da mascara e o caminho para salvar a imagem. 
# dessatiracao_seletiva(/content/imagem.png , /content/mascara.png , /content/imagem_resultante.png)

def dessaturacao_seletiva(imagem_caminho, mascara_caminho, saida_caminho):
     # Carrega a imagem
    imagem = cv.imread(imagem_caminho)
    if imagem is None:
        raise FileNotFoundError(f"Imagem não encontrada: {imagem_caminho}")

    # Carrega a máscara
    mascara = cv.imread(mascara_caminho, cv.IMREAD_GRAYSCALE)
    if mascara is None:
        raise FileNotFoundError(f"Máscara não encontrada: {mascara_caminho}")

    # Converte a imagem original para tons de cinza
    imagem_cinza = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)

    # Converte a imagem cinza de volta
    imagem_cinza_colorida = cv.cvtColor(imagem_cinza, cv.COLOR_GRAY2BGR)

    # Combina a imagem colorida e a máscara
    resultado = np.where(mascara[:, :, None] > 0, imagem, imagem_cinza_colorida)

    # Salva o resultado
    cv.imwrite(saida_caminho, resultado)
    print(f"Imagem salva em: {saida_caminho}")