from __future__ import annotations

import importlib
import inspect
import tempfile
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import numpy as np
import cv2 as cv

DEBUG_OPS = True  

def _pil_to_bgr(img_any):
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

    if arr.ndim == 2:
        return cv.cvtColor(arr, cv.COLOR_GRAY2BGR)

    if arr.ndim == 3 and arr.shape[2] == 3:
        try:
            return cv.cvtColor(arr, cv.COLOR_RGB2BGR)
        except Exception:
            return arr

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
    if isinstance(result, np.ndarray):
        return {"img": result}

    if isinstance(result, (str, Path)):
        bgr = _path_to_bgr(result)
        return {"img": bgr}

    if isinstance(result, dict):
        for key in ("img", "image", "output", "out", "result"):
            if key in result:
                val = result[key]
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
        out = {"img": None}
        out.update({k: v for k, v in result.items() if k not in ("img", "image", "output", "out", "result")})
        return out

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

    bgr = _pil_to_bgr(result)
    if isinstance(bgr, np.ndarray):
        return {"img": bgr}

    return {"img": None}


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
    "procimg.funcoes.mapear_cores",
    ["mapear_cores", "mapeamento_cores", "apply_lut"],
)

isolar_cor_fn = _import_any(
    "procimg.funcoes.isolar_cor",
    ["isolar_cor", "isolamento_de_cor", "isolamento_cor"],
)

realcar_cor_fn = _import_any(
    "procimg.funcoes.realcar_cor",
    ["realcar_cor", "realce_cor", "realce_de_cor"],
)

dessaturar_cor_seletiva_fn = _import_any(
    "procimg.funcoes.dessaturar_cor_seletiva",
    ["dessaturar_cor_seletiva", "dessaturacao_seletiva", "dessaturar_seletivo", "dessaturacao_selectiva"],
)

substituir_cor_fn = _import_any(
    "procimg.funcoes.substituir_cor",
    ["substituir_cor", "substituicao_cor", "replace_color", "trocar_cor"],
)

mudar_hue_fn = _import_any(
    "procimg.funcoes.mudar_hue",
    ["mudar_hue", "mudar_hue", "mudanca_de_hue", "shift_hue"],
)

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

def _call(fn, img, **params):
    if fn is None:
        return _fallback_passthrough(img, **params)

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

    with tempfile.TemporaryDirectory() as td:
        tmp_in = Path(td) / "in.png"
        cv.imwrite(str(tmp_in), img)

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
                    if out.get("img") is None and tmp_out and tmp_out.exists():
                        out = {"img": _path_to_bgr(tmp_out)}
                    if out.get("img") is not None:
                        return out
                    passed = True
            except TypeError:
                continue
            except Exception as e:
                if DEBUG_OPS:
                    print(f"[ops] chamada com caminho via '{name}' falhou:", e)
                passed = True

        if not passed:
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

    return _fallback_passthrough(img, **params)


def _call_substituir_cor(fn, img, **params):
    if fn is None:
        return _fallback_passthrough(img, **params)

    cor_origem = params.get("origem")
    cor_destino = params.get("destino")

    def bgr_to_rgb(t):
        return (t[2], t[1], t[0]) if isinstance(t, (tuple, list)) and len(t) == 3 else t

    out = _call(fn, img, **params)
    if out.get("img") is not None:
        mad = float(np.mean(np.abs(out["img"].astype(np.int16) - img.astype(np.int16))))
        if mad >= 0.5:
            return out

    params_rgb = params.copy()
    params_rgb["origem"] = bgr_to_rgb(cor_origem)
    params_rgb["destino"] = bgr_to_rgb(cor_destino)
    return _call(fn, img, **params_rgb)

def _equaliza_canais_core(
    img_bgr: "np.ndarray",
    space: str = "lab",          
    metodo: str = "clahe",       
    canais=None,                
    clip: float = 3.0,          
    tiles: int = 8,              
) -> dict:
    s = (space or "lab").lower()

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

    if work.ndim == 2:
        chs = [work]
    else:
        chs = [work[..., 0], work[..., 1], work[..., 2]]

    if canais is None:
        canais = list(range(len(chs)))

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

def _h_to_nome(h: int) -> str:
    pal = [
        ("vermelho", 0), ("laranja", 15), ("amarelo", 30), ("verde", 60),
        ("ciano", 90), ("azul", 120), ("roxo", 150), ("magenta", 165), ("rosa", 170),
    ]
    return min(pal, key=lambda kv: min(abs(kv[1]-h), 180-abs(kv[1]-h)))[0]

def _bgr_to_h(bgr: tuple[int,int,int]) -> int:
    patch = np.uint8([[bgr]]) 
    hsv = cv.cvtColor(patch, cv.COLOR_BGR2HSV)
    return int(hsv[0,0,0])

def _separar_canais(img_bgr: np.ndarray) -> dict:
    rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    lab = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)

    def put_label_outside(tile: np.ndarray, text: str, position: str = "bottom") -> np.ndarray:
        h, w = tile.shape[:2]
        bar_h = max(24, h // 10)
        bar = np.zeros((bar_h, w, 3), dtype=np.uint8) 

        font = cv.FONT_HERSHEY_SIMPLEX
        scale = max(1.2, min(2.5, w / 300.0))
        thickness = max(2, int(scale * 4))
        (tw, th), base = cv.getTextSize(text, font, scale, thickness)

        tx = max(4, (w - tw) // 2)
        ty = max(th + 4, (bar_h + th) // 2)  
        cv.putText(bar, text, (tx, ty), font, scale, (255, 255, 255), thickness, cv.LINE_AA)

        if position == "top":
            return np.vstack([bar, tile])
        return np.vstack([tile, bar])  

    def vis(img_space: np.ndarray, labels: tuple[str, str, str]) -> np.ndarray:
        ch = cv.split(img_space)
        tiles = []
        for c, lab_txt in zip(ch, labels):
            c8 = cv.normalize(c, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
            tile = cv.cvtColor(c8, cv.COLOR_GRAY2BGR)
            tiles.append(put_label_outside(tile, lab_txt, position="bottom"))
        return np.hstack(tiles)

    row_rgb = vis(rgb, ("R", "G", "B"))
    row_hsv = vis(hsv, ("H", "S", "V"))
    row_lab = vis(lab, ("L", "A", "B"))

    w = max(row_rgb.shape[1], row_hsv.shape[1], row_lab.shape[1])
    h = max(row_rgb.shape[0], row_hsv.shape[0], row_lab.shape[0])
    fit = lambda im: cv.resize(im, (w, h), interpolation=cv.INTER_AREA)

    grid = np.vstack([fit(row_rgb), fit(row_hsv), fit(row_lab)])
    return {"img": grid}

def _compara_canais_core(img_bgr: np.ndarray, space: str = "rgb", bins: int = 64) -> dict:
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
        ranges = [(0, 180), (0, 256), (0, 256)]  
        colors = ["m", "g", "k"]  
    elif s == "lab":
        work = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
        labels = ("L", "A", "B")
        ranges = [(0, 256), (0, 256), (0, 256)]
        colors = ["k", "c", "y"]
    else:
        return {"img": img_bgr}

    chs = [work[..., 0], work[..., 1], work[..., 2]]

    fig = _plt.figure(figsize=(8, 6), dpi=140)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], hspace=0.35, wspace=0.25)

    ax0 = fig.add_subplot(gs[0, 0])
    prev_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    ax0.imshow(prev_rgb)
    ax0.set_title("Preview", fontsize=10)
    ax0.axis("off")

    for i, (ch, lab, rng, col) in enumerate(zip(chs, labels, ranges, colors), start=1):
        r_idx = 0 if i == 1 else 1
        c_idx = 1 if i == 1 else (0 if i == 2 else 1)  
        ax = fig.add_subplot(gs[r_idx, c_idx])
        ax.hist(ch.ravel(), bins=int(bins), range=rng, color=col, alpha=0.85)
        ax.set_title(f"Canal {lab}", fontsize=10)
        ax.set_xlim(rng)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        ax.tick_params(labelsize=8)

    buf = _io.BytesIO()
    _plt.tight_layout()
    fig.savefig(buf, format="png", dpi=140)
    _plt.close(fig)
    buf.seek(0)
    data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    png_bgr = cv.imdecode(data, cv.IMREAD_COLOR)  
    return {"img": png_bgr}

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

    headers = ["Espaço", "Canal", "Média", "Desvio", "Mín", "Máx", "Mediana"]
    n = max(1, len(rows))
    fig_h = 1.0 + 0.28 * n  
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

def _graficos_dispersao_core(
    img_bgr: np.ndarray,
    space: str = "hsv",
    pairs: list[str] | None = None,
    sample: int = 20,         
    max_points: int = 50000,   
    alpha: float = 0.4,        
) -> dict:
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
        ranges = [(0, 180), (0, 256), (0, 256)]  
    elif s == "lab":
        work = cv.cvtColor(img_bgr, cv.COLOR_BGR2LAB)
        labels = ("L", "A", "B")
        default_pairs = ["L×A", "L×B", "A×B"]
        ranges = [(0, 256), (0, 256), (0, 256)]
    else:
        return {"img": img_bgr}

    pairs = pairs or default_pairs
    idx = {labels[0]: 0, labels[1]: 1, labels[2]: 2}

    h, w = work.shape[:2]
    total = h * w
    frac = max(1, min(100, int(sample))) / 100.0
    n = min(int(total * frac), int(max_points))

    rs = np.random.default_rng(12345)  
    ys = rs.integers(0, h, size=n, endpoint=False)
    xs = rs.integers(0, w, size=n, endpoint=False)

    ch0 = work[..., 0][ys, xs].astype(np.float32)
    ch1 = work[..., 1][ys, xs].astype(np.float32)
    ch2 = work[..., 2][ys, xs].astype(np.float32)
    CH = [ch0, ch1, ch2]

    k = min(3, len(pairs))
    fig_w = 4 * k + 2  
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
        ax.set_xlim(ranges[a_i])
        ax.set_ylim(ranges[b_i])
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        ax.set_title(f"{s.upper()} — {a_lab}×{b_lab}", fontsize=10)

    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    _plt.close(fig)
    buf.seek(0)
    data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    png_bgr = cv.imdecode(data, cv.IMREAD_COLOR)
    return {"img": png_bgr}

def _calcular_variacoes_core(
    img_a_bgr: np.ndarray,              
    img_b_bgr: np.ndarray,              
    space: str = "hsv",
    channels: list[str] | None = None,  
    make_heatmaps: bool = True,
    colormap: str = "TURBO",            
) -> dict:
    import matplotlib.pyplot as _plt
    import io as _io

    if img_a_bgr is None or img_b_bgr is None:
        return {"img": img_a_bgr} 

    if img_a_bgr.dtype != np.uint8:
        img_a_bgr = img_a_bgr.astype(np.uint8)
    if img_b_bgr.dtype != np.uint8:
        img_b_bgr = img_b_bgr.astype(np.uint8)

    hA, wA = img_a_bgr.shape[:2]
    hB, wB = img_b_bgr.shape[:2]
    if (hA, wA) != (hB, wB):
        inter = cv.INTER_AREA if (hB * wB) > (hA * wA) else cv.INTER_LINEAR
        img_b_bgr = cv.resize(img_b_bgr, (wA, hA), interpolation=inter)

    s = (space or "hsv").lower()
    if s == "rgb":
        A = cv.cvtColor(img_a_bgr, cv.COLOR_BGR2RGB)
        B = cv.cvtColor(img_b_bgr, cv.COLOR_BGR2RGB)
        labels = ("R", "G", "B")
        ranges = [(0, 256), (0, 256), (0, 256)]
    elif s == "lab":
        A = cv.cvtColor(img_a_bgr, cv.COLOR_BGR2LAB)
        B = cv.cvtColor(img_b_bgr, cv.COLOR_BGR2LAB)
        labels = ("L", "A", "B")
        ranges = [(0, 256), (0, 256), (0, 256)]
    else:
        A = cv.cvtColor(img_a_bgr, cv.COLOR_BGR2HSV)
        B = cv.cvtColor(img_b_bgr, cv.COLOR_BGR2HSV)
        labels = ("H", "S", "V")
        ranges = [(0, 180), (0, 256), (0, 256)]

    if channels:
        channels = [c.upper() for c in channels if c.upper() in labels]
        if not channels:
            channels = list(labels)
    else:
        channels = list(labels)

    chA = [A[..., 0], A[..., 1], A[..., 2]]
    chB = [B[..., 0], B[..., 1], B[..., 2]]

    name_to_idx = {lab: i for i, lab in enumerate(labels)}
    rows = []
    heat_imgs_rgb = []

    cv_cmap = getattr(cv, f"COLORMAP_{colormap.upper()}", cv.COLORMAP_TURBO)

    for lab in channels:
        i = name_to_idx[lab]
        a = chA[i].astype(np.int16)
        b = chB[i].astype(np.int16)

        if s == "hsv" and lab == "H":
            diff = np.abs(a - b)
            diff = np.minimum(diff, 180 - diff) 
            signed = (b - a)
            absdiff = diff.astype(np.float32)
            maxv = 90.0
        else:
            d = (b - a).astype(np.int16)
            signed = d
            absdiff = np.abs(d).astype(np.float32)
            maxv = 255.0

        mean_signed = float(np.mean(signed))
        mean_abs = float(np.mean(absdiff))
        vmin = float(np.min(absdiff))
        vmax = float(np.max(absdiff))
        rows.append([lab, f"{mean_signed:+.2f}", f"{mean_abs:.2f}", f"{vmin:.2f}", f"{vmax:.2f}"])

        if make_heatmaps:
            norm = np.clip((absdiff * (255.0 / max(1e-6, maxv))), 0, 255).astype(np.uint8)
            cm = cv.applyColorMap(norm, cv_cmap)
            heat_imgs_rgb.append(cv.cvtColor(cm, cv.COLOR_BGR2RGB))

    n_hmaps = len(heat_imgs_rgb) if make_heatmaps else 0
    fig_w = 8 if n_hmaps <= 1 else (12 if n_hmaps == 2 else 16)
    fig_h = 3.0 + (3.2 if n_hmaps else 0.6)

    fig = _plt.figure(figsize=(fig_w, fig_h), dpi=140)
    gs = fig.add_gridspec(2 if n_hmaps else 1, max(1, n_hmaps), hspace=0.35, wspace=0.25)

    ax_tbl = fig.add_subplot(gs[0, :]) if n_hmaps else fig.add_subplot(gs[0])
    ax_tbl.axis("off")
    headers = ["Canal", "Δ médio (com sinal)", "|Δ| médio", "min |Δ|", "max |Δ|"]
    tbl = ax_tbl.table(cellText=rows, colLabels=headers, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)
    ax_tbl.set_title(f"Variações — espaço {s.upper()}", fontsize=11, pad=8)

    if n_hmaps:
        for j in range(n_hmaps):
            ax = fig.add_subplot(gs[1, j])
            ax.imshow(heat_imgs_rgb[j])
            ax.axis("off")
            ax.set_title(f"Heatmap |Δ{channels[j]}|", fontsize=10, pad=4)

    buf = _io.BytesIO()
    _plt.tight_layout()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", pad_inches=0.2)
    _plt.close(fig)
    buf.seek(0)
    data = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    png_bgr = cv.imdecode(data, cv.IMREAD_COLOR)
    return {"img": png_bgr}
    
OPS = {
    "mapear-cores": lambda img, a: _call(
        mapear_cores_fn or _fallback_mapear_cores,
        img,
        nome_lut=getattr(a, "lut", "VIRIDIS"),
    ),

    "isolamento-cor": lambda img, a: _call(
        isolar_cor_fn or _fallback_passthrough,
        img,
        cor=_h_to_nome(_bgr_to_h(getattr(a, "cor", (0, 255, 0)))),
        tolerancia_h=int(getattr(a, "tol", 20)),
        s_min=60, v_min=50,
    ),

    "realce-cor": lambda img, a: _call(
        realcar_cor_fn or _fallback_passthrough,
        img,
        ganho_s=float(getattr(a, "ganho", 1.5)),
        ganho_v=1.0,
    ),

    "dessaturacao-seletiva": lambda img, a: _call(
        dessaturar_cor_seletiva_fn or _fallback_passthrough,
        img,
        cor=_h_to_nome(_bgr_to_h(getattr(a, "cor", (0, 0, 255)))),
        tol_h=12, s_min=60, v_min=50,
        s_bg=int(255 * float(getattr(a, "fator", 0.5))),
    ),

    "substituicao-cor": lambda img, a: _call(
        substituir_cor_fn or _fallback_passthrough,
        img,
        cor_original=_h_to_nome(_bgr_to_h(getattr(a, "cor_origem", (255, 0, 0)))),
        cor_nova=_h_to_nome(_bgr_to_h(getattr(a, "cor_destino", (0, 0, 255)))),
        tolerancia_h=int(getattr(a, "tol", 15)),
        s_min=60, v_min=50,
        alpha=1.0,
    ),

    "mudar-hue": lambda img, a: _call(
        mudar_hue_fn or _fallback_passthrough,
        img,
        deslocamento_hue=int(getattr(a, "hue", 30)),
    ),

    "equaliza-canais": lambda img, a: _equaliza_canais_core(
        img,
        space=getattr(a, "space", "lab"),      
        metodo=getattr(a, "metodo", "clahe"),   
        canais=getattr(a, "canais", None),      
        clip=getattr(a, "clip", 3.0),
        tiles=getattr(a, "tiles", 8),
    ),

    "separar-canais": lambda img, a: _separar_canais(img),

    "compara-canais": lambda img, a: _compara_canais_core(
        img,
        space=getattr(a, "space", "rgb"),
        bins=int(getattr(a, "bins", 64)),
    ),

    "calcular-estatisticas": lambda img, a: _calcular_estatisticas_core(
        img,
        spaces=getattr(a, "spaces", ["rgb", "hsv", "lab"]),
    ),

    "graficos-dispersao": lambda img, a: _graficos_dispersao_core(
        img,
        space=getattr(a, "space", "hsv"),
        pairs=getattr(a, "pairs", None),
        sample=int(getattr(a, "sample", 20)),
        max_points=int(getattr(a, "max_points", 50000)),
        alpha=float(getattr(a, "alpha", 0.4)),
    ),

    "calcular-variacoes": lambda img, a: _calcular_variacoes_core(
        img,
        getattr(a, "img_ref", img),                    
        space=getattr(a, "space", "hsv"),
        channels=getattr(a, "channels", ["H", "S", "V"]),
        make_heatmaps=bool(getattr(a, "heatmaps", True)),
        colormap=getattr(a, "colormap", "TURBO"),
    ),
}

from typing import Any as _Any


def _dessat_kwargs_from_ui(a) -> dict[str, _Any]:
    cor_bgr = getattr(a, "cor", (0, 0, 255))
    intensidade = float(getattr(a, "intensidade", 0.6))
    cor_nome = _h_to_nome(_bgr_to_h(cor_bgr))
    return {"cor": cor_nome, "fator": intensidade}

OPS["dessaturacao-seletiva"] = lambda img, a: (
    {"img": img, "warning": "dessaturacao-seletiva: implementação não encontrada (fallback)"}
    if dessaturar_cor_seletiva_fn is None else
    _call(dessaturar_cor_seletiva_fn, img, **_dessat_kwargs_from_ui(a))
)
