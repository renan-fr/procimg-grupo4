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
      - dict com 'img' (inclusive aninhado) -> idem
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
        img_val = result.get("img")
        # desaninha {"img": {"img": ...}}
        if isinstance(img_val, dict) and "img" in img_val:
            img_val = img_val["img"]
        if isinstance(img_val, (str, Path)):
            return {"img": _path_to_bgr(img_val)}
        if isinstance(img_val, np.ndarray):
            return {"img": img_val}
        bgr = _pil_to_bgr(img_val)
        if isinstance(bgr, np.ndarray):
            return {"img": bgr}
        # sem imagem válida: devolve bruto pra debug
        return {"img": None, **{k: v for k, v in result.items() if k != "img"}}

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
    ["substituir_cor", "substituicao_cor", "replace_color"]
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


# ---------- Registro de operações ----------
OPS = {
    "mapear-cores":           lambda img, a: _call(mapear_cores_fn or _fallback_mapear_cores, img, nome_lut=a.lut),
    "isolamento-cor":         lambda img, a: _call(isolamento_de_cor_fn or _fallback_passthrough, img, cor=a.cor, tolerancia=a.tol),
    "realce-cor":             lambda img, a: _call(realcar_cor_fn or _fallback_passthrough, img, ganho=a.ganho),
    "dessaturacao-seletiva":  lambda img, a: _call(dessaturacao_seletiva_fn or _fallback_passthrough, img, cor=a.cor, fator=a.fator),
    "substituicao-cor":       lambda img, a: _call_substituir_cor(substituir_cor_fn, img, origem=a.cor_origem, destino=a.cor_destino, tol=a.tol),
    "mudar-hue":              lambda img, a: _call(mudar_hue_fn or _fallback_passthrough, img, deslocamento_hue=a.hue),
}
