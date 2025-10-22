# 🧠 Sistema de Análise de Padrões de Cores (ProcIMG - Grupo 4)
### Disciplina: Processamento de Imagens de Computação Gráfica — UNIT

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

---

## 🎯 Visão Geral
O **ProcIMG** é um sistema voltado para a **análise e extração de padrões visuais** em imagens digitais, com foco na **manipulação de canais de cor**.  
O projeto possibilita aplicar transformações como:

- 🎨 **Isolamento de cores**
- ✨ **Realce de saturação e brilho**
- 🔁 **Substituição e mudança de tonalidades**
- 🖤 **Dessaturação seletiva**
- 🌈 **Mapeamento de cores (LUTs)**

Essas operações permitem estudar, visualizar e comparar o impacto de diferentes transformações cromáticas em imagens.

> 💡 Aplicações: controle de qualidade industrial, realce de exames médicos e experimentação didática em disciplinas de visão computacional.

---

## 🧩 Bibliotecas Principais
- **NumPy** — operações matriciais e numéricas  
- **OpenCV (cv2)** — leitura, conversão e manipulação de imagens  
- **Matplotlib** — visualização e análise comparativa  
- **Pillow (PIL)** *(opcional)* — compatibilidade com múltiplos formatos  
- **scikit-image** *(opcional)* — métricas e filtros adicionais

---

## ⚙️ Execução Local

### ▶️ Modo Terminal / Colab
1. Clone ou baixe o repositório:  
   ```bash
   git clone https://github.com/<usuario>/procimg-grupo4.git
   cd procimg-grupo4
   ```
2. Instale as dependências:  
   ```bash
   pip install -r requirements.txt
   ```
3. Acesse o diretório principal:  
   ```bash
   cd src
   ```
4. Execute o script principal:  
   ```bash
   python procimg_cli.py
   ```

🧭 O menu interativo guiará as operações disponíveis (1–9).  
📂 As imagens resultantes serão salvas automaticamente em `./saidas`.

---

## 💻 Frontend (em desenvolvimento)
Interface web desenvolvida por **Rafael Passos Sampaio**, permitindo comparar a **imagem original × processada** e ajustar parâmetros de forma interativa.

---

## 👥 Equipe de Desenvolvimento

| Integrante | Função | Descrição |
|-------------|--------|------------|
| **Caio Felipe Honorato Góis** | Isolamento e Substituição de Cor | Máscaras e trocas seletivas de tonalidade. |
| **Ricardo Dias Xavier** | Realce de Cor | Ajuste de saturação e brilho (HSV). |
| **Lenio Macedo Moura Morais** | Mudança de Cor | Deslocamento de matiz (Hue) no canal H. |
| **Tágore Campos Paraizo** | Dessaturação Seletiva | Manter uma faixa de cor e neutralizar o restante. |
| **Renan Silva Ferreira** | Mapeamento de Cores + Documentação | Aplicação de LUTs (Look-Up Tables) e documentação técnica. |
| **Rafael Passos Sampaio** | Frontend / Interface | Interface web para integração visual do sistema. |

---

## 🖼️ Exemplos de Saída
*(Versão inicial — será atualizada com capturas reais nas próximas versões.)*

| Entrada | Resultado |
|----------|------------|
| ![Original](https://via.placeholder.com/150) | ![Processada](https://via.placeholder.com/150) |

---

## 🏗️ Estrutura do Projeto
```bash
📂 procimg-grupo4/
 ┣ 📁 src/
 ┃ ┣ procimg_cli.py         → versão de linha de comando (CLI)
 ┃ ┣ frontend/              → interface web (em desenvolvimento)
 ┃ ┣ utils/                 → funções auxiliares
 ┣ 📁 saidas/               → imagens resultantes das operações
 ┣ README.md                → documentação geral do projeto
 ┣ requirements.txt         → dependências necessárias
```

---

## 📚 Créditos
Projeto desenvolvido pelo **Grupo 4** da disciplina de **Processamento de Imagens de Computação Gráfica** —  
**Universidade Tiradentes (UNIT)**, 2025.  
**Orientação:** Profª **Layse Santos Souza**.
