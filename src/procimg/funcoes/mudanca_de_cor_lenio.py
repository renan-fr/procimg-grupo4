# Função Mudança de Cor - Lenio Morais

# Função para isolar e mudar o Hue da imagem selecionada pelo 
# valor especificado

# Para chamar a função só precisamos entregar a imagem e o número do Hue adicionado
# a função é chamada assim mudar_hue(imagemSelecionada, hueAdicionado)

def mudar_hue(img_bgr, deslocamento_hue):
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    h = (h.astype(np.int32) + deslocamento_hue) % 180
    h = h.astype(np.uint8)
    hsv_modificado = cv.merge([h, s, v])

    img_bgr_modificada = cv.cvtColor(hsv_modificado, cv.COLOR_HSV2BGR)
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title(f"Hue +{deslocamento_hue}")
    plt.imshow(cv.cvtColor(img_bgr_modificada, cv.COLOR_BGR2RGB))
    plt.axis("off")

    plt.show()

