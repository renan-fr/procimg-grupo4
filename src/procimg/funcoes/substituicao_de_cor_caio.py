import cv2 as cv
import numpy as np
import matplotlib.pylab as plt
def trocar_cor(cor_original='vermelho', cor_nova='azul', tolerancia=50):
    """
    Troca uma cor específica por outra na imagem mantendo o restante original.
    
    Parâmetros:
    - cor_original: string com a cor que será trocada
    - cor_nova: string com a cor que substituirá
    - tolerancia: valor de tolerância para detecção da cor (0-255)
    
    Cores disponíveis:
    vermelho, verde, azul, amarelo, ciano, magenta, laranja, roxo, rosa, branco, preto
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
    cores_mask = {
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
    
    # Define os valores RGB para cada cor
    cores_rgb = {
        'vermelho': [255, 0, 0],
        'verde': [0, 255, 0],
        'azul': [0, 0, 255],
        'amarelo': [255, 255, 0],
        'ciano': [0, 255, 255],
        'magenta': [255, 0, 255],
        'laranja': [255, 165, 0],
        'roxo': [128, 0, 128],
        'rosa': [255, 192, 203],
        'branco': [255, 255, 255],
        'preto': [0, 0, 0]
    }
    
    # Verifica se as cores são válidas
    if cor_original.lower() not in cores_mask:
        print(f"\nCor original '{cor_original}' não reconhecida!")
        print("\nCORES DISPONÍVEIS:")
        print("vermelho, verde, azul, amarelo, ciano, magenta,")
        print("laranja, roxo, rosa, branco, preto")
        return None
    
    if cor_nova.lower() not in cores_rgb:
        print(f"\nCor nova '{cor_nova}' não reconhecida!")
        print("\nCORES DISPONÍVEIS:")
        print("vermelho, verde, azul, amarelo, ciano, magenta,")
        print("laranja, roxo, rosa, branco, preto")
        return None
    
    # Aplica a máscara para detectar a cor original
    mask = cores_mask[cor_original.lower()]
    
    # Cria resultado mantendo a imagem original
    resultado = img_rgb.copy()
    
    # Troca apenas os pixels da cor original pela cor nova
    resultado[mask] = cores_rgb[cor_nova.lower()]
    
    # Exibe imagens
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(img_rgb)
    plt.title("Imagem Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(resultado)
    plt.title(f"Troca: {cor_original} -> {cor_nova}")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()
    
    return resultado

trocar_cor('azul', 'vermelho', tolerancia=50)
