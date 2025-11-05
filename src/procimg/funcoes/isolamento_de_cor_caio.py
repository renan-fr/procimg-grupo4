import cv2 as cv
import numpy as np
import matplotlib.pylab as plt
def isolar_cor(cor_escolhida='vermelho', tolerancia=50):
    """
    Isola uma cor específica usando separação de canais RGB.
    
    Parâmetros:
    - cor_escolhida: string com o nome da cor ('vermelho', 'verde', 'azul', 'amarelo', 
                     'ciano', 'magenta', 'laranja', 'roxo', 'rosa', 'branco', 'preto')
    - tolerancia: valor de tolerância para o isolamento (0-255)
    
    Retorna:
    - Exibe a imagem original e o resultado do isolamento
    """
    
    # Upload da imagem
    print("Faça upload da sua imagem:")
    uploaded = files.upload()
    
    # Carrega a imagem
    img_name = list(uploaded.keys())[0]
    img_array = np.frombuffer(uploaded[img_name], dtype=np.uint8)
    img_bgr = cv.imdecode(img_array, cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    
    # Separa os canais RGB
    (R, G, B) = cv.split(img_rgb)
    
    # Define as máscaras para cada cor
    cores = {
        'vermelho': (R > 150 - tolerancia) & (G < 100 + tolerancia) & (B < 100 + tolerancia),
        'verde': (G > 150 - tolerancia) & (R < 100 + tolerancia) & (B < 100 + tolerancia),
        'azul': (B > 150 - tolerancia) & (R < 100 + tolerancia) & (G < 100 + tolerancia),
        'amarelo': (R > 150 - tolerancia) & (G > 150 - tolerancia) & (B < 100 + tolerancia),
        'ciano': (G > 150 - tolerancia) & (B > 150 - tolerancia) & (R < 100 + tolerancia),
        'magenta': (R > 150 - tolerancia) & (B > 150 - tolerancia) & (G < 100 + tolerancia),
        'laranja': (R > 200 - tolerancia) & (G > 100 - tolerancia//2) & (G < 180 + tolerancia//2) & (B < 80 + tolerancia),
        'roxo': (R > 100 - tolerancia) & (B > 150 - tolerancia) & (G < 100 + tolerancia) & (R < B),
        'rosa': (R > 180 - tolerancia) & (G > 100 - tolerancia) & (B > 100 - tolerancia) & (R > G) & (R > B),
        'branco': (R > 200 - tolerancia) & (G > 200 - tolerancia) & (B > 200 - tolerancia),
        'preto': (R < 50 + tolerancia) & (G < 50 + tolerancia) & (B < 50 + tolerancia)
    }
    
    # Verifica se a cor é válida
    if cor_escolhida.lower() not in cores:
        print(f"\nCor '{cor_escolhida}' não reconhecida!")
        print("\nCORES DISPONÍVEIS:")
        print("vermelho, verde, azul, amarelo, ciano, magenta,")
        print("laranja, roxo, rosa, branco, preto")
        return None
    
    # Aplica a máscara
    mask = cores[cor_escolhida.lower()]
    resultado = img_rgb.copy()
    resultado[~mask] = [0, 0, 0]  # Pixels que não são da cor ficam pretos
    
    # Exibe imagens
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(img_rgb)
    plt.title("Imagem Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(resultado)
    plt.title(f"Cor isolada: {cor_escolhida}")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    return resultado
