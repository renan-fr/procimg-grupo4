# 🧠 Sistema de Análise de Padrões de Cores (ProcIMG - Grupo 4)
### Disciplina: Processamento de Imagens de Computação Gráfica — UNIT

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![Status](https://img.shields.io/badge/Status-Funcional%20%2F%20CLI%20Pronto-green)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

---

## 🎯 Visão Geral
O **ProcIMG** é um sistema voltado para a **análise e transformação de cores em imagens digitais**, com foco no estudo e aplicação prática de técnicas de **processamento de imagem**.  
O projeto permite aplicar e comparar diferentes manipulações cromáticas por meio de uma interface de linha de comando simples e intuitiva (CLI).

Principais operações:
- 🎨 **Isolamento de cores**
- ✨ **Realce de saturação e brilho**
- 🔁 **Substituição e mudança de tonalidades**
- 🖤 **Dessaturação seletiva**
- 🌈 **Mapeamento de cores (LUTs)**

Essas transformações possibilitam explorar o comportamento dos canais de cor (RGB, HSV, LAB) e visualizar os impactos visuais de cada operação.

> 💡 Aplicações: controle de qualidade industrial, realce de exames médicos, análises visuais e experimentação didática em disciplinas de visão computacional.

---

## 🧩 Bibliotecas Principais
- **NumPy** — operações matriciais e numéricas  
- **OpenCV (cv2)** — leitura, conversão e manipulação de imagens  
- **Matplotlib** — visualização e análise comparativa  
- **Pillow (PIL)** — compatibilidade com múltiplos formatos  
- **scikit-image** — filtros e métricas complementares  
- **Typer + Rich** — criação de interface de linha de comando moderna

---

## ⚙️ Execução Local (CLI)

### 1. Clonar e preparar o ambiente
```bash
git clone https://github.com/renan-fr/procimg-grupo4.git
cd procimg-grupo4

# criar ambiente virtual (recomendado)
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate     # Windows

# instalar o projeto em modo editável
pip install -e .
```

---

### 2. Estrutura recomendada de pastas
```bash
📂 procimg-grupo4/
 ┣ 📁 entradas/     → imagens originais (entrada)
 ┣ 📁 saidas/       → resultados processados (gerados automaticamente)
 ┣ 📁 src/procimg/
 ┃ ┣ cli.py        → interface de linha de comando (Typer)
 ┃ ┗ ops.py        → operações de processamento
 ┣ pyproject.toml
 ┗ README.md
```

Crie as pastas se ainda não existirem:
```bash
mkdir -p entradas saidas
```

---

### 3. Ver operações disponíveis
```bash
procimg ops
```

Exemplo de saída:
```
     Operações disponíveis
┏━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃  #  ┃ op                    ┃
┡━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│  1  │ dessaturacao-seletiva │
│  2  │ isolamento-cor        │
│  3  │ mapear-cores          │
│  4  │ mudar-hue             │
│  5  │ realce-cor            │
│  6  │ substituicao-cor      │
└─────┴───────────────────────┘
```

---

### 4. Executar uma operação

Coloque uma imagem dentro da pasta `entradas/` (por exemplo, `imagem_teste.jpg`)  
e rode o comando com o nome da operação desejada.

#### Exemplo 1 — mudar o matiz (Hue)
```bash
procimg run --op mudar-hue --in imagem_teste.jpg --param hue=25
```
➡️ Resultado salvo automaticamente em `saidas/imagem_teste__mudar-hue.png`

#### Exemplo 2 — aplicar mapeamento de cores (LUT)
```bash
procimg run --op mapear-cores --in imagem_teste.jpg --param lut=TURBO
```

#### Exemplo 3 — realçar saturação
```bash
procimg run --op realce-cor --in imagem_teste.jpg --param ganho=1.5
```

#### Exemplo 4 — dessaturar mantendo apenas tons de vermelho
```bash
procimg run --op dessaturacao-seletiva --in imagem_teste.jpg --param cor=vermelho
```

---

### 🧭 Dicas úteis
- Você pode passar vários parâmetros:
  ```bash
  procimg run --op substituicao-cor --in flor.jpg --param cor-origem=vermelho --param cor-destino=azul
  ```
- O parâmetro `--out` é opcional. Se omitido, o resultado vai para `saidas/<nome>__<op>.png`.
- O nome da imagem pode ser só o arquivo (ex: `flor.jpg`) — o CLI automaticamente procura em `entradas/`.

---

## 💻 Frontend (em desenvolvimento)
Interface web desenvolvida por **Rafael Passos Sampaio**, que permitirá visualizar e comparar a imagem original e a processada lado a lado, com ajuste interativo de parâmetros.  
Essa interface será implementada em **Streamlit** e usará as mesmas funções do CLI.

---

## 👥 Equipe de Desenvolvimento

| Integrante | Função | Descrição |
|-------------|--------|------------|
| **Caio Felipe Honorato Góis** | Isolamento e Substituição de Cor | Máscaras e trocas seletivas de tonalidade. |
| **Ricardo Dias Xavier** | Realce de Cor | Ajuste de saturação e brilho (HSV). |
| **Lenio Macedo Moura Morais** | Mudança de Cor | Deslocamento de matiz (Hue) no canal H. |
| **Tágore Campos Paraizo** | Dessaturação Seletiva | Manter uma faixa de cor e neutralizar o restante. |
| **Renan Silva Ferreira** | Mapeamento de Cores + Documentação | Aplicação de LUTs (Look-Up Tables) e documentação técnica. |
| **Rafael Passos Sampaio** | Frontend / Interface | Interface web (Streamlit) e integração visual do sistema. |

---

## 🖼️ Exemplo de Resultado

| Entrada | Resultado (`mudar-hue` com `hue=25`) |
|----------|--------------------------------------|
| ![Original](https://via.placeholder.com/250x150.png?text=Original) | ![Processada](https://via.placeholder.com/250x150.png?text=Hue+25) |

---

## 📚 Créditos
Projeto desenvolvido pelo **Grupo 4** da disciplina de **Processamento de Imagens de Computação Gráfica** —  
**Universidade Tiradentes (UNIT)**, 2025.  
**Orientação:** Profª **Layse Santos Souza**.
