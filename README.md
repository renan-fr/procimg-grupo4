# Sistema de Análise e Manipulação de Canais de Cor (ProcIMG - Grupo 4)
### Disciplina: Processamento de Imagens de Computação Gráfica — UNIT

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Interface](https://img.shields.io/badge/Interface-Streamlit-green)
![CLI](https://img.shields.io/badge/CLI-Typer%20%2F%20Rich-yellow)
![Status](https://img.shields.io/badge/Status-Funcional%20%2F%20Completo-brightgreen)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

---

## 🎯 Visão Geral

O **Sistema de Análise e Manipulação de Canais de Cor** é um sistema completo para **análise, visualização e transformação de canais de cor em imagens digitais**, combinando:

- **Interface Web (Streamlit)** para experimentação visual  
- **CLI (Typer + Rich)** para automação e análises rápidas  
- **Pacote Python instalável (pyproject)**, com funções modulares em `src/procimg/funcoes/`  
- **Núcleo unificado (`ops.py`)** que integra todas as operações

O foco do projeto é ser **didático, modular e experimental**, permitindo estudar:

- Comportamento dos canais RGB / HSV / LAB  
- Mapeamento de cores (LUTs)  
- Realce, equalização e manipulação seletiva  
- Análises comparativas entre imagens  
- Heatmaps, histogramas, dispersões e estatísticas  

---

## 🧩 Tecnologias Utilizadas

| Categoria | Tecnologias |
|----------|-------------|
| Processamento de Imagens | **OpenCV**, **NumPy**, **Pillow**, **scikit-image** |
| Visualização | **Matplotlib**, **Streamlit** |
| Terminal / CLI | **Typer**, **Rich** |
| Organização | Estrutura modular em `src/procimg/` + `pyproject.toml` |

---

## 📁 Estrutura do Projeto

```
procimg-grupo4/
 ┣ entradas/               # imagens de entrada (exemplos para testes)
 ┣ saidas/                 # resultados gerados pela CLI ou Streamlit
 ┣ docs/                   # documentos, relatórios e material adicional
 ┣ demo/                   # vídeo de apresentação do sistema
 ┣ app.py                  # interface Streamlit
 ┣ pyproject.toml          # define dependências e instalação com pip
 ┗ src/
    ┗ procimg/
       ┣ cli.py            # CLI Typer + Rich
       ┣ ops.py            # dispatcher central (todas as operações)
       ┗ funcoes/          # funções individuais
          ┣ mapear_cores.py
          ┣ isolar_cor.py
          ┣ realcar_cor.py
          ┣ dessaturar_cor_seletiva.py
          ┣ substituir_cor.py
          ┣ mudar_hue.py
          ┣ equalizar_canais.py
          ┣ separar_canais.py
          ┣ comparar_canais.py
          ┣ calcular_estatisticas.py
          ┣ gerar_grafico_dispersao.py
          ┗ calcular_variacoes.py
```

---

## ⚙️ Instalação Local (recomendada)

A instalação é via **modo editável**, usando o `pyproject.toml`.

### 1. Criar ambiente virtual (opcional, mas recomendado)

```bash
python -m venv .venv
```

Ativar:

- **Windows (PowerShell)**  
  ```bash
  .venv\Scripts\Activate.ps1
  ```

- **Linux/Mac**  
  ```bash
  source .venv/bin/activate
  ```

---

### 2. Instalar o projeto e dependências

Na raiz do projeto:

```bash
pip install -e .
```

Isso instala:

- todas as dependências principais  
- Streamlit  
- o pacote `procimg-grupo4`  
- o módulo CLI (usável via `python -m procimg.cli`)  

---

## 💻 Executar a Interface Web (Streamlit)

```bash
python -m streamlit run app.py
```

O navegador abrirá automaticamente em:  
http://localhost:8501

---

## 🧪 Executar via CLI (Linha de Comando)

Listar operações:

```bash
python -m procimg.cli ops
```

Executar operação:

```bash
python -m procimg.cli run --op mapear-cores --in imagem_mapear_cores.jpg
```

---

## 👥 Equipe de Desenvolvimento

| Integrante | Função | Descrição |
|-------------|--------|------------|
| **Caio Felipe Honorato Góis** | Isolamento e Substituição de Cor | Máscaras e trocas seletivas de tonalidade. |
| **Lenio Macedo Moura Morais** | Mudança de Cor | Deslocamento de matiz (Hue). |
| **Renan Silva Ferreira** | Mapeamento de Cores, Equalização e Interface Web | LUTs, equalização, Streamlit e integração. |
| **Ricardo Dias Xavier** | Realce de Cor | Ajuste de saturação e brilho. |
| **Tágore Campos Paraizo** | Dessaturação Seletiva | Manter uma faixa de cor e neutralizar o restante. |
| **Todos** | Documentação e Revisão | Revisão técnica e organização.

---

## 📚 Créditos

Projeto desenvolvido pelo **Grupo 4** da disciplina de **Processamento de Imagens de Computação Gráfica** — UNIT, 2025.  
Orientação: **Profª Layse Santos Souza**.

**Integrantes do Grupo 4:**
- **Caio Felipe Honorato Góis**
- **Lenio Macedo Moura Morais**
- **Renan Silva Ferreira**
- **Ricardo Dias Xavier**
- **Tágore Campos Paraizo**