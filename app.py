# app.py — ProcIMG (Streamlit)

from pathlib import Path
from types import SimpleNamespace
import io

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --- Imports do pacote ---
try:
    from procimg import ops as ops_mod
except Exception as e:
    st.error(f"Erro ao importar procimg.ops: {e}")
    st.stop()

# =============================
# Diretórios e LUTs
# =============================
ENTRADAS_DIR = Path("entradas")
SAIDAS_DIR = Path("saidas")
SAIDAS_DIR.mkdir(parents=True, exist_ok=True)

CV_LUTS = [
    "VIRIDIS", "PLASMA", "INFERNO", "MAGMA", "CIVIDIS",
    "TURBO", "JET", "HSV", "HOT", "BONE", "RAINBOW"
]

# =============================
# Funções auxiliares
# =============================

def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (0, 0, 0)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)

def ensure_dict(result):
    return result if isinstance(result, dict) else {"img": result}

def read_image_any(input_bytes: bytes | None, path: Path | None):
    if input_bytes:
        arr = np.frombuffer(input_bytes, dtype=np.uint8)
        img = cv.imdecode(arr, cv.IMREAD_COLOR)
        return img, "upload"
    if path and path.exists():
        img = cv.imread(str(path), cv.IMREAD_COLOR)
        return img, str(path)
    return None, ""

def show_image(title: str, img_bgr: np.ndarray):
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    st.image(img_rgb, caption=title, use_container_width=True)

def pil_to_bgr(img_any):
    try:
        from PIL import Image
        if isinstance(img_any, Image.Image):
            arr = np.array(img_any)
        else:
            arr = img_any

        if arr is None:
            return None
        if isinstance(arr, np.ndarray):
            if arr.ndim == 2:
                return cv.cvtColor(arr, cv.COLOR_GRAY2BGR)
            if arr.ndim == 3 and arr.shape[2] == 3:
                return cv.cvtColor(arr, cv.COLOR_RGB2BGR)
            if arr.ndim == 3 and arr.shape[2] == 4:
                rgb = cv.cvtColor(arr, cv.COLOR_RGBA2RGB)
                return cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
            return arr
        return None
    except Exception:
        return None

def first_ndarray(*vals):
    for v in vals:
        if v is None:
            continue
        if isinstance(v, np.ndarray):
            return v
        bgr = pil_to_bgr(v)
        if isinstance(bgr, np.ndarray):
            return bgr
    return None

# =============================
# Visualização de LUTs
# =============================

def _cv_colormap(lut_name: str) -> int:
    return getattr(cv, f"COLORMAP_{lut_name.upper()}", cv.COLORMAP_VIRIDIS)

def _render_lut_gradient(lut_name: str, width=512, height=30):
    grad = np.linspace(0, 255, width, dtype=np.uint8)
    grad = np.tile(grad, (height, 1))
    mapped = cv.applyColorMap(grad, _cv_colormap(lut_name))
    return cv.cvtColor(mapped, cv.COLOR_BGR2RGB)

def show_luts_gradients():
    """Exibe barras de gradiente para todos os LUTs do CV_LUTS."""
    items = [(lut, _render_lut_gradient(lut)) for lut in CV_LUTS]
    fig, axs = plt.subplots(len(items), 1, figsize=(6, 0.35 * len(items)))
    if len(items) == 1:
        axs = [axs]
    for ax, (name, img_rgb) in zip(axs, items):
        ax.imshow(img_rgb)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    st.image(buf, caption="Comparativo de LUTs (gradientes)")
    plt.close(fig)

def show_luts_pair(lut_a: str, lut_b: str, img_bgr: np.ndarray | None):
    """Mostra gradientes e aplicação dos LUTs A e B."""
    ga = _render_lut_gradient(lut_a)
    gb = _render_lut_gradient(lut_b)
    st.image([ga, gb],
             caption=[f"{lut_a} (gradiente)", f"{lut_b} (gradiente)"],
             use_container_width=True)

    if img_bgr is not None:
        gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
        a = cv.applyColorMap(gray, _cv_colormap(lut_a))
        b = cv.applyColorMap(gray, _cv_colormap(lut_b))
        a_rgb = cv.cvtColor(a, cv.COLOR_BGR2RGB)
        b_rgb = cv.cvtColor(b, cv.COLOR_BGR2RGB)
        st.image([a_rgb, b_rgb],
                 caption=[f"{lut_a} (na imagem)", f"{lut_b} (na imagem)"],
                 use_container_width=True)

# =============================
# Interface
# =============================

st.set_page_config(page_title="ProcIMG — Streamlit", layout="wide")
st.title("ProcIMG — Análise de Cores (Streamlit)")

left, right = st.columns([1, 1])

# ---------- Painel de entrada ----------
with left:
    st.subheader("Entrada")
    source = st.radio("Fonte da imagem", ["Upload", "Pasta 'entradas/'"], horizontal=True)
    uploaded = None
    selected_path = None

    if source == "Upload":
        file = st.file_uploader("Envie uma imagem (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if file is not None:
            uploaded = file.read()
    else:
        ENTRADAS_DIR.mkdir(exist_ok=True)
        imgs = sorted([p for p in ENTRADAS_DIR.glob("*.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        name_map = {p.name: p for p in imgs}
        choice = st.selectbox("Selecione um arquivo em 'entradas/'", ["(nenhum)"] + list(name_map.keys()))
        if choice != "(nenhum)":
            selected_path = name_map[choice]

    img, origem = read_image_any(uploaded, selected_path)
    if img is None:
        st.info("Carregue uma imagem ou escolha um arquivo em 'entradas/'.")
    else:
        show_image("Imagem de entrada", img)

# ---------- Painel de operação ----------
with right:
    st.subheader("Operação")
    ops_keys = list(ops_mod.OPS.keys())
    if not ops_keys:
        st.error("Nenhuma operação registrada em ops.OPS.")
        st.stop()

    op = st.selectbox("Escolha a operação", ops_keys)
    a = SimpleNamespace()

    if op == "mapear-cores":
        a.lut = st.selectbox("LUT", CV_LUTS, index=0)

        # Botão de visualização dos LUTs
        if st.button("Visualizar todos os LUTs"):
            show_luts_gradients()

        # Comparar dois LUTs
        c1, c2 = st.columns(2)
        with c1:
            lut_a = st.selectbox("Comparar A", CV_LUTS,
                                 index=CV_LUTS.index("TURBO") if "TURBO" in CV_LUTS else 0,
                                 key="lut_a")
        with c2:
            lut_b = st.selectbox("Comparar B", CV_LUTS,
                                 index=CV_LUTS.index("JET") if "JET" in CV_LUTS else 1,
                                 key="lut_b")

        if st.button("Comparar LUTs (gradiente + imagem)"):
            show_luts_pair(lut_a, lut_b, img)

    elif op == "isolamento-cor":
        cor_hex = st.color_picker("Cor alvo", value="#00FF00")
        a.cor = hex_to_bgr(cor_hex)
        a.tol = st.slider("Tolerância", 0, 100, 20)

    elif op == "realce-cor":
        a.ganho = st.slider("Ganho", 0.1, 5.0, 1.5, step=0.1)

    elif op == "dessaturacao-seletiva":
        cor_hex = st.color_picker("Cor alvo", value="#00FF00")
        a.cor = hex_to_bgr(cor_hex)
        a.intensidade = st.slider("Intensidade (PB do fundo)", 0.0, 1.0, 1.0, step=0.05)
        st.caption("Dica: escolha uma cor presente (ex.: verde das árvores). Intensidade 1.0 = PB forte no fundo.")

    elif op == "substituicao-cor":
        origem_hex = st.color_picker("Cor de origem", value="#00FFFF")
        destino_hex = st.color_picker("Cor destino", value="#FF00FF")
        a.cor_origem = hex_to_bgr(origem_hex)
        a.cor_destino = hex_to_bgr(destino_hex)
        a.tol = st.slider("Tolerância", 0, 100, 15)

    elif op == "mudar-hue":
        a.hue = st.slider("Deslocamento de Hue (OpenCV 0–179)", 0, 179, 30)
    else:
        st.warning("Operação sem UI dedicada — nenhum parâmetro adicional.")

    # Executar operação selecionada
    run_it = st.button("▶️ Executar")

    if run_it:
        if img is None:
            st.error("Selecione/Carregue uma imagem antes de executar.")
        else:
            try:
                out_raw = ops_mod.OPS[op](img, a)
                out_dict = ensure_dict(out_raw)
                candidatos = [
                    out_dict.get("img"),
                    out_dict.get("image"),
                    out_dict.get("output"),
                    out_dict.get("out"),
                    out_dict.get("result"),
                ]
                if isinstance(out_raw, (tuple, list)):
                    candidatos.extend(list(out_raw))
                out_img = first_ndarray(*candidatos)

                if out_img is None or not isinstance(out_img, np.ndarray):
                    st.error("A operação não retornou uma imagem válida.")
                    st.write({
                        "tipo_retorno": type(out_raw).__name__,
                        "chaves_detectadas": list(out_dict.keys())
                    })
                    st.stop()

                if img.shape == out_img.shape and img.dtype == out_img.dtype:
                    mad = float(np.mean(np.abs(out_img.astype(np.int16) - img.astype(np.int16))))
                    if mad < 0.5:
                        st.warning(
                            "A saída parece idêntica à entrada. Ajuste parâmetros (ex.: tolerância/ganho) "
                            "ou verifique se a função está em fallback."
                        )

                st.success("Operação concluída.")
                show_image(f"Saída — {op}", out_img)

                stem = "upload" if origem == "upload" else Path(origem).stem
                out_path = SAIDAS_DIR / f"{stem}__{op}.png"
                cv.imwrite(str(out_path), out_img)

                _, buf = cv.imencode(".png", out_img)
                st.download_button(
                    label="Baixar imagem processada (.png)",
                    data=buf.tobytes(),
                    file_name=out_path.name,
                    mime="image/png",
                )
                st.caption(f"Arquivo salvo em: {out_path}")
            except Exception as e:
                st.exception(e)

st.divider()
st.markdown(
    "Dica: Coloque imagens em **entradas/** para testá-las rapidamente. Saídas vão para **saidas/** automaticamente."
)
