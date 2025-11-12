# 🧠 Sistema de Análise e Manipulação de Canais de Cor (ProcIMG - Grupo 4)
### Disciplina: Processamento de Imagens de Computação Gráfica — UNIT

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![Interface](https://img.shields.io/badge/Interface-Streamlit-green)
![Status](https://img.shields.io/badge/Status-Funcional%20%2F%20Completo-brightgreen)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

---

## 🎯 Visão Geral

O **ProcIMG** é um sistema completo de **análise e transformação de cores em imagens digitais**, combinando uma **interface web interativa (Streamlit)** e uma **CLI (Typer)**.  
Permite explorar e comparar técnicas de **processamento de imagem** aplicadas aos canais de cor (RGB, HSV, LAB), com foco didático e experimental.

O sistema permite:

- 🎨 **Isolamento e substituição de cores**
- ✨ **Realce de saturação e brilho**
- 🖤 **Dessaturação seletiva**
- 🌈 **Mapeamento de cores (LUTs OpenCV)**
- ⚙️ **Equalização de canais (CLAHE / Histograma)**
- 🔍 **Separação de canais e visualização em grid**

> 💡 Ideal para experimentos visuais, ensino de visão computacional e análises cromáticas (ex: realce, contraste e distribuição de cores).

---

## 🧩 Tecnologias Utilizadas

| Categoria | Ferramentas |
|------------|-------------|
| Processamento | **OpenCV**, **NumPy**, **Matplotlib**, **Pillow** |
| Visualização | **Streamlit** (interface interativa), **Matplotlib** |
| CLI / Terminal | **Typer**, **Rich** |
| Organização | Estrutura modular em `src/procimg/` |

---

## ⚙️ Execução Local (modo CLI)

### 1. Clonar e preparar o ambiente

```bash
git clone https://github.com/renan-fr/procimg-grupo4.git
cd procimg-grupo4

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate     # Windows

pip install -e .
```

---

### 2. Estrutura recomendada

```bash
📂 procimg-grupo4/
 ┣ 📁 entradas/     → imagens de entrada
 ┣ 📁 saidas/       → resultados gerados automaticamente
 ┣ 📁 src/procimg/
 ┃ ┣ cli.py         → interface de linha de comando (Typer)
 ┃ ┣ app.py         → interface web (Streamlit)
 ┃ ┗ ops.py         → núcleo de operações e integração
 ┣ pyproject.toml
 ┗ README.md
```

Se ainda não existirem:
```bash
mkdir -p entradas saidas
```

---

### 3. Ver operações disponíveis

```bash
procimg ops
```

Saída esperada:

```
┏━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃  #  ┃ op                    ┃
┡━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│  1  │ mapear-cores          │
│  2  │ isolamento-cor        │
│  3  │ realce-cor            │
│  4  │ dessaturacao-seletiva │
│  5  │ substituicao-cor      │
│  6  │ mudar-hue             │
│  7  │ equaliza-canais       │
│  8  │ separar-canais        │
└─────┴───────────────────────┘
```

---

### 4. Executar uma operação via CLI

```bash
procimg run --op equaliza-canais --in imagem_teste.jpg --param space=lab --param metodo=clahe
```
➡️ Resultado salvo automaticamente em `saidas/imagem_teste__equaliza-canais.png`

---

## 💻 Execução via Interface Web (Streamlit)

Além do CLI, o **ProcIMG** conta com uma **interface gráfica completa** em **Streamlit**, que permite visualizar e comparar resultados lado a lado, ajustando os parâmetros de cada operação em tempo real.

### 1. Iniciar o servidor local

```bash
streamlit run app.py
```

### 2. Acessar no navegador

Abra o endereço exibido no terminal, normalmente:  
👉 [http://localhost:8501](http://localhost:8501)

### 3. Como usar

1. Escolha uma imagem de **upload** ou selecione uma da pasta `entradas/`.  
2. No painel **Operação**, selecione a técnica desejada:  
   - `mapear-cores` → aplicar LUTs do OpenCV  
   - `isolamento-cor` → destacar faixa de cor  
   - `realce-cor` → aumentar saturação/brilho  
   - `dessaturacao-seletiva` → manter cor específica  
   - `substituicao-cor` → trocar uma cor por outra  
   - `mudar-hue` → deslocar matiz global  
   - `equaliza-canais` → equalizar contraste em RGB/HSV/LAB  
   - `separar-canais` → exibir canais lado a lado (R/G/B, H/S/V ou L/A/B)

---

## 👥 Equipe de Desenvolvimento

| Integrante | Função | Descrição |
|-------------|--------|------------|
| **Caio Felipe Honorato Góis** | Isolamento e Substituição de Cor | Máscaras e trocas seletivas de tonalidade. |
| **Lenio Macedo Moura Morais** | Mudança de Cor | Deslocamento de matiz (Hue) no canal H. |
| **Renan Silva Ferreira** | Mapeamento de Cores, Equalização e Interface Web | Aplicação de LUTs, implementação da interface Streamlit e integração geral do sistema. |
| **Ricardo Dias Xavier** | Realce de Cor | Ajuste de saturação e brilho (HSV). |
| **Tágore Campos Paraizo** | Dessaturação Seletiva | Manter uma faixa de cor e neutralizar o restante. |
| **Todos os integrantes** | Documentação e Revisão Técnica | Contribuição coletiva na escrita, revisão e padronização dos materiais do projeto. 

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
