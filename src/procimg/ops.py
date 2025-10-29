import importlib

def _optional_import(module_path, func_name):
    try:
        mod = importlib.import_module(module_path)
        fn = getattr(mod, func_name, None)
        return fn if callable(fn) else None
    except Exception:
        return None

def _ensure_dict(result):
    return result if isinstance(result, dict) else {"img": result}

mapear_cores_fn         = _optional_import("procimg.funcoes.mapeamento_cores_renan", "mapear_cores")
isolamento_de_cor_fn    = _optional_import("procimg.funcoes.isolamento_de_cor_caio", "isolamento_de_cor")
realcar_cor_fn          = _optional_import("procimg.funcoes.realce_de_cor_ricardo", "realcar_cor")
dessaturar_seletivo_fn  = _optional_import("procimg.funcoes.dessaturacao_seletiva_tagore", "dessaturar_seletivo")
substituir_cor_fn       = _optional_import("procimg.funcoes.substituicao_de_cor_caio", "substituir_cor")
mudar_hue_fn            = _optional_import("procimg.funcoes.mudanca_de_cor_lenio", "mudar_hue")

def _fallback_passthrough(img, **kwargs): return {"img": img}

def _fallback_mapear_cores(img, nome_lut="VIRIDIS", **kwargs):
    import cv2 as cv
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    lut = getattr(cv, f"COLORMAP_{nome_lut.upper()}", cv.COLORMAP_VIRIDIS)
    return {"img": cv.applyColorMap(gray, lut)}

OPS = {
    "mapear-cores": lambda img, a: _ensure_dict((mapear_cores_fn or _fallback_mapear_cores)(img, nome_lut=a.lut)),
    "isolamento-cor": lambda img, a: _ensure_dict((isolamento_de_cor_fn or _fallback_passthrough)(img, cor=a.cor, tolerancia=a.tol)),
    "realce-cor": lambda img, a: _ensure_dict((realcar_cor_fn or _fallback_passthrough)(img, ganho=a.ganho)),
    "dessaturacao-seletiva": lambda img, a: _ensure_dict((dessaturar_seletivo_fn or _fallback_passthrough)(img, cor=a.cor, fator=a.fator)),
    "substituicao-cor": lambda img, a: _ensure_dict((substituir_cor_fn or _fallback_passthrough)(img, origem=a.cor_origem, destino=a.cor_destino, tol=a.tol)),
    "mudar-hue": lambda img, a: _ensure_dict((mudar_hue_fn or _fallback_passthrough)(img, deslocamento_hue=a.hue)),
}
