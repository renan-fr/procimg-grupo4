# src/procimg/ops.py
from __future__ import annotations

import importlib
import inspect
import tempfile
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import numpy as np
import cv2 as cv

DEBUG_OPS = True  # deixe True por enquanto para ver logs no terminal

# ---------- util: normalização de imagens ----------
def _pil_to_bgr(img_any):
    """Converte PIL.Image ou arrays RGB/GRAY/RGBA para BGR uint8 (OpenCV)."""
    try:
        from PIL import Image
        if isinstance(img_any, Image.Image):
            arr = np.array(img_any)
        else:
            arr = img_any
    except Exception:
        arr = img_any

    if arr is None:
        return None
    if not isinstance(arr, np.ndarray):
        return None

    # GRAY -> BGR
    if arr.ndim == 2:
        return cv.cvtColor(arr, cv.COLOR_GRAY2BGR)

    # RGB -> BGR (heurística)
    if arr.ndim == 3 and arr.shape[2] == 3:
        try:
            return cv.cvtColor(arr, cv.COLOR_RGB2BGR)
        except Exception:
            return arr

    # RGBA -> BGR
    if arr.ndim == 3 and arr.shape[2] == 4:
        try:
            rgb = cv.cvtColor(arr, cv.COLOR_RGBA2RGB)
            return cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
        except Exception:
            return arr

    return arr


def _path_to_bgr(p: str | Path | None):
    if not p:
        return None
    try:
        img = cv.imread(str(p), cv.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _normalize_output(result: Any) -> dict[str, Any]:
    """
    Normaliza o retorno:
      - ndarray -> {'img': ndarray}
      - PIL.Image -> {'img': ndarray BGR}
      - str/path -> lê com cv.imread -> {'img': ndarray}
      - dict -> aceita chaves 'img' | 'image' | 'output' | 'out' | 'result'
      - tuple/list -> pega o primeiro item útil
    """
    # ndarray
    if isinstance(result, np.ndarray):
        return {"img": result}

    # path/str
    if isinstance(result, (str, Path)):
        bgr = _path_to_bgr(result)
        return {"img": bgr}

    # dict
    if isinstance(result, dict):
        # tenta várias chaves usuais
        for key in ("img", "image", "output", "out", "result"):
            if key in result:
                val = result[key]
                # desaninha {'image': {'image': ...}} etc.
                if isinstance(val, dict):
                    for subk in ("img", "image", "output", "out", "result"):
                        if subk in val:
                            val = val[subk]
                            break
                if isinstance(val, (str, Path)):
                    return {"img": _path_to_bgr(val)}
                if isinstance(val, np.ndarray):
                    return {"img": val}
                bgr = _pil_to_bgr(val)
                if isinstance(bgr, np.ndarray):
                    return {"img": bgr}
        # sem imagem válida: devolve bruto p/ debug
        out = {"img": None}
        out.update({k: v for k, v in result.items() if k not in ("img", "image", "output", "out", "result")})
        return out

    # tuple/list
    if isinstance(result, (tuple, list)):
        for item in result:
            if isinstance(item, np.ndarray):
                return {"img": item}
            if isinstance(item, (str, Path)):
                return {"img": _path_to_bgr(item)}
            bgr = _pil_to_bgr(item)
            if isinstance(bgr, np.ndarray):
                return {"img": bgr}
        return {"img": None}

    # PIL ou outros
    bgr = _pil_to_bgr(result)
    if isinstance(bgr, np.ndarray):
        return {"img": bgr}

    return {"img": None}


# ---------- import com nomes alternativos ----------
def _import_any(module_path: str, candidate_names: list[str]):
    try:
        mod = importlib.import_module(module_path)
    except Exception as e:
        if DEBUG_OPS:
            print(f"[ops] Falha ao importar módulo {module_path}: {e}")
        return None

    for name in candidate_names:
        fn = getattr(mod, name, None)
        if callable(fn):
            if DEBUG_OPS:
                print(f"[ops] OK: {module_path}.{name}")
            return fn

    if DEBUG_OPS:
        print(f"[ops] Nenhuma função candidata encontrada em {module_path}: {candidate_names}")
    return None


mapear_cores_fn = _import_any(
    "procimg.funcoes.mapeamento_cores_renan",
    ["mapear_cores", "mapeamento_cores", "apply_lut"]
)

isolamento_de_cor_fn = _import_any(
    "procimg.funcoes.isolamento_de_cor_caio",
    ["isolamento_de_cor", "isolar_cor", "isolamento_cor"]
)

realcar_cor_fn = _import_any(
    "procimg.funcoes.realce_de_cor_ricardo",
    ["realcar_cor", "realce_cor", "realce_de_cor"]
)

dessaturacao_seletiva_fn = _import_any(
    "procimg.funcoes.dessaturacao_seletiva_tagore",
    ["dessaturacao_seletiva", "dessaturar_seletivo", "dessaturacao_selectiva"]
)

substituir_cor_fn = _import_any(
    "procimg.funcoes.substituicao_de_cor_caio",
    ["substituir_cor", "substituicao_cor", "replace_color", "trocar_cor"]
)

mudar_hue_fn = _import_any(
    "procimg.funcoes.mudanca_de_cor_lenio",
    ["mudar_hue", "mudanca_de_hue", "shift_hue", "mudar_cor"]
)


# ---------- fallbacks ----------
def _fallback_passthrough(img, **kwargs):
    if DEBUG_OPS:
        print(f"[ops] Fallback PASSTHROUGH. Parâmetros: {kwargs}")
    return {"img": img}


def _fallback_mapear_cores(img, nome_lut="VIRIDIS", **kwargs):
    try:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        lut = getattr(cv, f"COLORMAP_{nome_lut.upper()}", cv.COLORMAP_VIRIDIS)
        return {"img": cv.applyColorMap(gray, lut)}
    except Exception:
        return {"img": img}


# ---------- caller inteligente ----------
def _call(fn, img, **params):
    """
    Adapta para funções que esperam:
      - ndarray diretamente: fn(img, ...)
      - caminho de entrada: fn(img_path, ...)
      - saída obrigatória: fn(..., saida_caminho=...)
      - assinatura tipo CLI: fn(img, args_namespace)
    Depois normaliza retorno para {'img': ndarray BGR}.
    """
    if fn is None:
        return _fallback_passthrough(img, **params)

    # 1) Tenta assinatura direta (img, **params)
    try:
        result = fn(img, **params)
        out = _normalize_output(result)
        if out.get("img") is not None:
            return out
    except TypeError:
        pass
    except Exception as e:
        if DEBUG_OPS:
            print("[ops] chamada (img, **params) falhou:", e)

    # 2) Tenta assinatura estilo Namespace (img, args)
    try:
        result = fn(img, SimpleNamespace(**params))
        out = _normalize_output(result)
        if out.get("img") is not None:
            return out
    except TypeError:
        pass
    except Exception as e:
        if DEBUG_OPS:
            print("[ops] chamada (img, Namespace) falhou:", e)

    # 3) Tenta assinatura com caminho: fn(img_path, **params)
    #    -> escrevemos imagem em arquivo temporário
    with tempfile.TemporaryDirectory() as td:
        tmp_in = Path(td) / "in.png"
        cv.imwrite(str(tmp_in), img)

        # se função exigir saida_caminho, vamos criar
        sig = None
        try:
            sig = inspect.signature(fn)
        except Exception:
            pass

        needs_out = False
        out_param_names = {"saida_caminho", "saida", "out_path", "output_path", "destino"}
        if sig is not None:
            for pname in out_param_names:
                if pname in sig.parameters:
                    needs_out = True
                    break

        tmp_out = Path(td) / "out.png" if needs_out else None

        # prepara kwargs, mapeando possíveis nomes de entrada
        kw = params.copy()
        in_param_names = ["img_path", "caminho_imagem", "entrada", "input_path", "imagem_path", "path"]
        passed = False
        for name in in_param_names:
            try:
                if sig is None or name in sig.parameters:
                    kw[name] = str(tmp_in)
                    if tmp_out:
                        for oname in out_param_names:
                            if sig is None or oname in sig.parameters:
                                kw[oname] = str(tmp_out)
                                break
                    result = fn(**kw)
                    out = _normalize_output(result)
                    # se a função não retornar imagem mas escreveu arquivo, lê-lo
                    if out.get("img") is None and tmp_out and tmp_out.exists():
                        out = {"img": _path_to_bgr(tmp_out)}
                    if out.get("img") is not None:
                        return out
                    passed = True
            except TypeError:
                # assinatura diferente; tenta próximo nome
                continue
            except Exception as e:
                if DEBUG_OPS:
                    print(f"[ops] chamada com caminho via '{name}' falhou:", e)
                passed = True

        if not passed:
            # última tentativa: posição simples (fn(tmp_in, ...))
            try:
                result = fn(str(tmp_in), **params)
                out = _normalize_output(result)
                if out.get("img") is None and tmp_out and tmp_out.exists():
                    out = {"img": _path_to_bgr(tmp_out)}
                if out.get("img") is not None:
                    return out
            except Exception as e:
                if DEBUG_OPS:
                    print("[ops] chamada posicional com caminho falhou:", e)

    # 4) último recurso
    return _fallback_passthrough(img, **params)


# ---------- wrappers específicos (quando útil) ----------
def _call_substituir_cor(fn, img, **params):
    """
    Algumas implementações assumem RGB. Tentamos BGR e RGB.
    Esperados: origem/destino (tuplas) e tol/tolerancia.
    """
    if fn is None:
        return _fallback_passthrough(img, **params)

    cor_origem = params.get("origem")
    cor_destino = params.get("destino")

    def bgr_to_rgb(t):
        return (t[2], t[1], t[0]) if isinstance(t, (tuple, list)) and len(t) == 3 else t

    # 1) tenta BGR
    out = _call(fn, img, **params)
    if out.get("img") is not None:
        # se muito parecido, tenta RGB
        mad = float(np.mean(np.abs(out["img"].astype(np.int16) - img.astype(np.int16))))
        if mad >= 0.5:
            return out

    # 2) tenta RGB
    params_rgb = params.copy()
    params_rgb["origem"] = bgr_to_rgb(cor_origem)
    params_rgb["destino"] = bgr_to_rgb(cor_destino)
    return _call(fn, img, **params_rgb)

# ------------------------------------------------------------
# equaliza-canais (núcleo) — retorna {'img': BGR}
# ------------------------------------------------------------
def _equaliza_canais_core(
    img_bgr: "np.ndarray",
    space: str = "lab",          # "rgb" | "hsv" | "lab"
    metodo: str = "clahe",       # "clahe" | "hist"
    canais=None,                 # None => todos; ex.: [0], [0,1]
    clip: float = 3.0,           # CLAHE
    tiles: int = 8,              # CLAHE (tiles x tiles)
) -> dict:
    """
    Equaliza contraste por canal no espaço escolhido.
    Retorna {'img': ndarray BGR} para encaixar no app atual.
    """
    s = (space or "lab").lower()

    # BGR -> espaço alvo
    if s == "rgb":
        work = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    elif s == "hsv":
        work = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    elif s == "lab":
        work = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
    else:
        if DEBUG_OPS:
            print(f"[equaliza-canais] espaço inválido: {space}. Mantendo BGR.")
        return {"img": img_bgr}

    # separa canais
    if work.ndim == 2:
        chs = [work]
    else:
        chs = [work[..., 0], work[..., 1], work[..., 2]]

    if canais is None:
        canais = list(range(len(chs)))

    # equalização
    metodo = (metodo or "clahe").lower()
    if metodo == "hist":
        for i in canais:
            chs[i] = cv.equalizeHist(chs[i])
    elif metodo == "clahe":
        clahe = cv.createCLAHE(
            clipLimit=float(clip if clip is not None else 3.0),
            tileGridSize=(int(tiles or 8), int(tiles or 8)),
        )
        for i in canais:
            chs[i] = clahe.apply(chs[i])
    else:
        if DEBUG_OPS:
            print(f"[equaliza-canais] método desconhecido: {metodo}. Sem alteração.")

    # junta e volta pra BGR
    if len(chs) == 1:
        merged = chs[0]
        if s == "rgb":
            out_bgr = cv.cvtColor(merged, cv.COLOR_RGB2BGR)
        elif s == "hsv":
            out_bgr = cv.cvtColor(merged, cv.COLOR_HSV2BGR)
        else:
            out_bgr = cv.cvtColor(merged, cv.COLOR_LAB2BGR)
        return {"img": out_bgr}

    work2 = np.stack(chs, axis=-1)
    if s == "rgb":
        out_bgr = cv.cvtColor(work2, cv.COLOR_RGB2BGR)
    elif s == "hsv":
        out_bgr = cv.cvtColor(work2, cv.COLOR_HSV2BGR)
    else:
        out_bgr = cv.cvtColor(work2, cv.COLOR_LAB2BGR)
    return {"img": out_bgr}

# ==== helpers p/ adaptar UI -> funções ====
def _h_to_nome(h: int) -> str:
    pal = [
        ("vermelho", 0), ("laranja", 15), ("amarelo", 30), ("verde", 60),
        ("ciano", 90), ("azul", 120), ("roxo", 150), ("magenta", 165), ("rosa", 170),
    ]
    return min(pal, key=lambda kv: min(abs(kv[1]-h), 180-abs(kv[1]-h)))[0]

def _bgr_to_h(bgr: tuple[int,int,int]) -> int:
    patch = np.uint8([[bgr]])  # (1,1,3) BGR
    hsv = cv.cvtColor(patch, cv.COLOR_BGR2HSV)
    return int(hsv[0,0,0])

# ---------- separar-canais ----------
def _separar_canais(img_bgr: np.ndarray) -> dict:
    """Gera uma grade 3x3 (RGB/HSV/LAB) com legenda fora do tile (topo/rodapé)."""
    rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)

    def put_label_outside(tile: np.ndarray, text: str, position: str = "bottom") -> np.ndarray:
        """Adiciona uma barra de legenda fora do tile (top/bottom), com texto centralizado."""
        h, w = tile.shape[:2]
        bar_h = max(24, h // 10)  # ~10% da altura, mínimo 24px
        bar = np.zeros((bar_h, w, 3), dtype=np.uint8)  # barra preta

        font = cv.FONT_HERSHEY_SIMPLEX
        scale = max(1.2, min(2.5, w / 300.0))
        thickness = max(2, int(scale * 4))
        (tw, th), base = cv.getTextSize(text, font, scale, thickness)

        tx = max(4, (w - tw) // 2)
        ty = max(th + 4, (bar_h + th) // 2)  # verticalmente centralizado na barra
        cv.putText(bar, text, (tx, ty), font, scale, (255, 255, 255), thickness, cv.LINE_AA)

        if position == "top":
            return np.vstack([bar, tile])
        return np.vstack([tile, bar])  # bottom

    def vis(img_space: np.ndarray, labels: tuple[str, str, str]) -> np.ndarray:
        ch = cv.split(img_space)
        tiles = []
        for c, lab_txt in zip(ch, labels):
            c8 = cv.normalize(c, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
            tile = cv.cvtColor(c8, cv.COLOR_GRAY2BGR)
            tiles.append(put_label_outside(tile, lab_txt, position="bottom"))
        # todas têm altura igual (tile + barra); empilha lado a lado
        return np.hstack(tiles)

    row_rgb = vis(rgb, ("R", "G", "B"))
    row_hsv = vis(hsv, ("H", "S", "V"))
    row_lab = vis(lab, ("L", "A", "B"))

    # garante mesmo tamanho entre linhas antes de empilhar
    w = max(row_rgb.shape[1], row_hsv.shape[1], row_lab.shape[1])
    h = max(row_rgb.shape[0], row_hsv.shape[0], row_lab.shape[0])
    fit = lambda im: cv.resize(im, (w, h), interpolation=cv.INTER_AREA)

    grid = np.vstack([fit(row_rgb), fit(row_hsv), fit(row_lab)])
    return {"img": grid}

# ---------- comparar-canais ----------
def _compara_canais_core(img_bgr: np.ndarray, space: str = "rgb", bins: int = 64) -> dict:
    """
    Gera uma figura com PREVIEW + 3 histogramas (um por canal) no espaço escolhido.
    Retorna {'img': ndarray BGR} para exibição/ download no app.
    """
    import matplotlib.pyplot as _plt
    import io as _io

    s = (space or "rgb").lower()
    if s == "rgb":
        work = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        labels = ("R", "G", "B")
        ranges = [(0, 256), (0, 256), (0, 256)]
        colors = ["r", "g", "b"]
    elif s == "hsv":
        work = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
        labels = ("H", "S", "V")
        ranges = [(0, 180), (0, 256), (0, 256)]  # OpenCV: H 0–179
        colors = ["m", "g", "k"]  # só pra diferenciar; não impacta o BGR final
    elif s == "lab":
        work = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
        labels = ("L", "A", "B")
        ranges = [(0, 256), (0, 256), (0, 256)]
        colors = ["k", "c", "y"]
    else:
        return {"img": img_bgr}

    chs = [work[..., 0], work[..., 1], work[..., 2]]

    # --- monta figura: 2x2 (preview + 3 histos) ---
    fig = _plt.figure(figsize=(8, 6), dpi=140)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], hspace=0.35, wspace=0.25)

    # Preview (canto superior esquerdo)
    ax0 = fig.add_subplot(gs[0, 0])
    # mostrar preview sempre em RGB
    prev_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    ax0.imshow(prev_rgb)
    ax0.set_title("Preview", fontsize=10)
    ax0.axis("off")

    # Histogramas (H1, H2, H3)
    for i, (ch, lab, rng, col) in enumerate(zip(chs, labels, ranges, colors), start=1):
        r_idx = 0 if i == 1 else 1
        c_idx = 1 if i == 1 else (0 if i == 2 else 1)  # coloca 1º hist em (0,1), 2º em (1,0), 3º em (1,1)
        ax = fig.add_subplot(gs[r_idx, c_idx])
        ax.hist(ch.ravel(), bins=int(bins), range=rng, color=col, alpha=0.85)
        ax.set_title(f"Canal {lab}", fontsize=10)
        ax.set_xlim(rng)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        ax.tick_params(labelsize=8)

    # Exporta figura para PNG e volta como BGR
    buf = _io.BytesIO()
    _plt.tight_layout()
    fig.savefig(buf, format="png", dpi=140)
    _plt.close(fig)
    buf.seek(0)
    data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    png_bgr = cv.imdecode(data, cv.IMREAD_COLOR)  # já vem BGR via OpenCV
    return {"img": png_bgr}

# ---------- calcular_estatisticas ----------
def _calcular_estatisticas_core(img_bgr: np.ndarray, spaces: list[str] | None = None) -> dict:
    """
    Gera uma imagem com uma tabela de estatísticas por canal (mean, std, min, max, median)
    para os espaços selecionados (RGB/HSV/LAB). Retorna {'img': ndarray BGR}.
    """
    import matplotlib.pyplot as _plt
    import io as _io

    spaces = spaces or ["rgb", "hsv", "lab"]
    rows = []

    def stats_of(ch: np.ndarray) -> tuple[float, float, int, int, float]:
        v = ch.ravel()
        return float(np.mean(v)), float(np.std(v)), int(np.min(v)), int(np.max(v)), float(np.median(v))

    for s in spaces:
        s_low = s.lower()
        if s_low == "rgb":
            work = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
            labs = ("R", "G", "B")
        elif s_low == "hsv":
            work = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
            labs = ("H", "S", "V")
        elif s_low == "lab":
            work = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
            labs = ("L", "A", "B")
        else:
            continue

        chs = [work[..., 0], work[..., 1], work[..., 2]]
        for lab, ch in zip(labs, chs):
            mean, std, vmin, vmax, med = stats_of(ch)
            rows.append([s_low.upper(), lab, f"{mean:.1f}", f"{std:.1f}", vmin, vmax, f"{med:.1f}"])

    # Desenha tabela como imagem
    headers = ["Espaço", "Canal", "Média", "Desvio", "Mín", "Máx", "Mediana"]
    n = max(1, len(rows))
    fig_h = 1.0 + 0.28 * n  # altura aproxima (linhas)
    fig = _plt.figure(figsize=(8, fig_h), dpi=140)
    ax = fig.add_subplot(111)
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=headers, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.2)
    ax.set_title("Estatísticas por Canal", fontsize=10, pad=8)

    buf = _io.BytesIO()
    _plt.tight_layout()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", pad_inches=0.2)
    _plt.close(fig)
    buf.seek(0)
    data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    png_bgr = cv.imdecode(data, cv.IMREAD_COLOR)
    return {"img": png_bgr}

# ---------- graficos-dispersao ----------
def _graficos_dispersao_core(
    img_bgr: np.ndarray,
    space: str = "hsv",
    pairs: list[str] | None = None,
    sample: int = 20,          # porcentagem da imagem a amostrar (1–100)
    max_points: int = 50000,   # teto absoluto de pontos
    alpha: float = 0.4,        # transparência dos pontos
) -> dict:
    """
    Gera uma figura com gráficos de dispersão para pares de canais do espaço escolhido.
    Suporta RGB, HSV, LAB. Retorna {'img': ndarray BGR}.
    """
    import matplotlib.pyplot as _plt
    import io as _io

    s = (space or "hsv").lower()
    if s == "rgb":
        work = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        labels = ("R", "G", "B")
        default_pairs = ["R×G", "R×B", "G×B"]
        ranges = [(0, 256), (0, 256), (0, 256)]
    elif s == "hsv":
        work = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
        labels = ("H", "S", "V")
        default_pairs = ["H×S", "H×V", "S×V"]
        ranges = [(0, 180), (0, 256), (0, 256)]  # H 0–179 no OpenCV
    elif s == "lab":
        work = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
        labels = ("L", "A", "B")
        default_pairs = ["L×A", "L×B", "A×B"]
        ranges = [(0, 256), (0, 256), (0, 256)]
    else:
        return {"img": img_bgr}

    pairs = pairs or default_pairs
    # mapeia nome -> índice
    idx = {labels[0]: 0, labels[1]: 1, labels[2]: 2}

    h, w = work.shape[:2]
    total = h * w
    frac = max(1, min(100, int(sample))) / 100.0
    n = min(int(total * frac), int(max_points))

    # amostragem aleatória
    rs = np.random.default_rng(12345)  # fixo p/ reprodutibilidade
    ys = rs.integers(0, h, size=n, endpoint=False)
    xs = rs.integers(0, w, size=n, endpoint=False)

    # pega canais amostrados
    ch0 = work[..., 0][ys, xs].astype(np.float32)
    ch1 = work[..., 1][ys, xs].astype(np.float32)
    ch2 = work[..., 2][ys, xs].astype(np.float32)
    CH = [ch0, ch1, ch2]

    # figura: 1..3 gráficos (até 3 pares)
    k = min(3, len(pairs))
    fig_w = 4 * k + 2  # preview opcional não usado aqui -> só os scatters
    fig = _plt.figure(figsize=(fig_w, 4), dpi=140)
    _plt.subplots_adjust(wspace=0.35, bottom=0.18, top=0.88)

    for i in range(k):
        p = pairs[i].upper().replace("X", "×")
        try:
            a_lab, b_lab = p.split("×")
            a_lab = a_lab.strip()
            b_lab = b_lab.strip()
        except Exception:
            continue
        if a_lab not in idx or b_lab not in idx:
            continue
        a_i, b_i = idx[a_lab], idx[b_lab]
        ax = fig.add_subplot(1, k, i + 1)
        ax.scatter(CH[a_i], CH[b_i], s=2, alpha=float(alpha), edgecolors="none")
        ax.set_xlabel(a_lab)
        ax.set_ylabel(b_lab)
        # limites coerentes com o espaço
        ax.set_xlim(ranges[a_i])
        ax.set_ylim(ranges[b_i])
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        ax.set_title(f"{s.upper()} — {a_lab}×{b_lab}", fontsize=10)

    # exporta a figura para PNG -> BGR
    import io as _io
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    _plt.close(fig)
    buf.seek(0)
    data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    png_bgr = cv.imdecode(data, cv.IMREAD_COLOR)
    return {"img": png_bgr}


# ---------- Registro de operações ----------
OPS = {
    # app: "mapear-cores" -> mapear_cores(img, nome_lut)
    "mapear-cores": lambda img, a: _call(
        mapear_cores_fn or _fallback_mapear_cores,
        img,
        nome_lut=getattr(a, "lut", "VIRIDIS"),
    ),

    # app: "isolamento-cor" -> isolamento_cor(img, cor="nome", tolerancia_h, s_min, v_min)
    # UI manda cor em BGR e tol 0–100
    "isolamento-cor": lambda img, a: _call(
        isolamento_de_cor_fn or _fallback_passthrough,
        img,
        cor=_h_to_nome(_bgr_to_h(getattr(a, "cor", (0,255,0)))),
        tolerancia_h=int(getattr(a, "tol", 20)),
        s_min=60, v_min=50,
    ),

    # app: "realce-cor" -> realce_cor(img, ganho_s, ganho_v)
    # UI tem um slider único "ganho" (S); V = 1.0
    "realce-cor": lambda img, a: _call(
        realcar_cor_fn or _fallback_passthrough,
        img,
        ganho_s=float(getattr(a, "ganho", 1.5)),
        ganho_v=1.0,
    ),

    # app: "dessaturacao-seletiva" -> dessaturacao_seletiva(img, cor="nome", tol_h, s_min, v_min, s_bg)
    # UI manda cor em BGR e "fator (0–1)" p/ fundo -> s_bg = 255*fator
    "dessaturacao-seletiva": lambda img, a: _call(
        dessaturacao_seletiva_fn or _fallback_passthrough,
        img,
        cor=_h_to_nome(_bgr_to_h(getattr(a, "cor", (0,0,255)))),
        tol_h=12, s_min=60, v_min=50,
        s_bg=int(255*float(getattr(a, "fator", 0.5))),
    ),

    # app: "substituicao-cor" -> trocar_cor(img, cor_original, cor_nova, tolerancia_h, s_min, v_min, alpha)
    # UI manda origem/destino em BGR e tol 0–100
    "substituicao-cor": lambda img, a: _call(
        substituir_cor_fn or _fallback_passthrough,
        img,
        cor_original=_h_to_nome(_bgr_to_h(getattr(a, "cor_origem", (255,0,0)))),
        cor_nova=_h_to_nome(_bgr_to_h(getattr(a, "cor_destino", (0,0,255)))),
        tolerancia_h=int(getattr(a, "tol", 15)),
        s_min=60, v_min=50,
        alpha=1.0,
    ),

    # app: "mudar-hue" -> mudar_hue(img, deslocamento_hue)
    "mudar-hue": lambda img, a: _call(
        mudar_hue_fn or _fallback_passthrough,
        img,
        deslocamento_hue=int(getattr(a, "hue", 30)),
    ),

    # equalização por canal (compatível com UI atual – 1 imagem)
    "equaliza-canais": lambda img, a: _equaliza_canais_core(
        img,
        space=getattr(a, "space", "lab"),       # padrão LAB
        metodo=getattr(a, "metodo", "clahe"),   # "clahe" | "hist"
        canais=getattr(a, "canais", None),      # None => todos
        clip=getattr(a, "clip", 3.0),
        tiles=getattr(a, "tiles", 8),
    ),

    # separa canais (RGB | HSV | LAB) em grade 1x3
    "separar-canais": lambda img, a: _separar_canais(img),

        "compara-canais": lambda img, a: _compara_canais_core(
        img,
        space=getattr(a, "space", "rgb"),
        bins=int(getattr(a, "bins", 64)),
    ),

    # estatísticas por canal (tabela-figura)
    "calcular-estatisticas": lambda img, a: _calcular_estatisticas_core(
        img,
        spaces=getattr(a, "spaces", ["rgb", "hsv", "lab"]),
    ),

    # gráficos de dispersão entre pares de canais
    "graficos-dispersao": lambda img, a: _graficos_dispersao_core(
        img,
        space=getattr(a, "space", "hsv"),
        pairs=getattr(a, "pairs", None),
        sample=int(getattr(a, "sample", 20)),
        max_points=int(getattr(a, "max_points", 50000)),
        alpha=float(getattr(a, "alpha", 0.4)),
    ),


}

# ---------- dessaturacao-seletiva (mapeamento limpo) ----------
from typing import Any as _Any


def _dessat_kwargs_from_ui(a) -> dict[str, _Any]:
    """
    Compatível com dessaturacao_seletiva(img_bgr, cor="vermelho", fator=0.5).
    - 'cor' (BGR da UI) -> nome aproximado via hue
    - 'intensidade' (0–1) na UI -> 'fator' (0–1) **direto**
    """
    cor_bgr = getattr(a, "cor", (0, 0, 255))
    intensidade = float(getattr(a, "intensidade", 0.6))
    cor_nome = _h_to_nome(_bgr_to_h(cor_bgr))
    return {"cor": cor_nome, "fator": intensidade}

OPS["dessaturacao-seletiva"] = lambda img, a: (
    {"img": img, "warning": "dessaturacao-seletiva: implementação não encontrada (fallback)"}
    if dessaturacao_seletiva_fn is None else
    _call(dessaturacao_seletiva_fn, img, **_dessat_kwargs_from_ui(a))
)
