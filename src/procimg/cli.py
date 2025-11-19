from __future__ import annotations

import cv2 as cv
import numpy as np
import typer
from pathlib import Path
from typing import Any, Callable
from rich import print
from rich.table import Table
import types

try:
    from . import ops as ops_mod  
except Exception as e:
    ops_mod = None
    _ops_import_err = e
else:
    _ops_import_err = None

app = typer.Typer(add_completion=False, help="PROCIMG — CLI para operações de cor (RGB/HSV/LAB)")

def _read(img_path: str) -> np.ndarray:
    p = Path(img_path)
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {img_path}")
    img = cv.imread(str(p), cv.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Falha ao ler imagem: {img_path}")
    return img

def _write(img: np.ndarray, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(out_path, img):
        raise RuntimeError(f"Falha ao salvar: {out_path}")

def _auto_cast(v: str) -> Any:
    s = v.strip()
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s

def _parse_params(kv_list: list[str] | None) -> dict[str, Any]:
    if not kv_list:
        return {}
    out: dict[str, Any] = {}
    for item in kv_list:
        if "=" not in item:
            raise typer.BadParameter(f"Parâmetro inválido (use k=v): {item}")
        k, v = item.split("=", 1)
        out[k.strip()] = _auto_cast(v)
    return out

def _resolve_dispatch() -> Callable[[str, np.ndarray, dict[str, Any]], np.ndarray]:
    if ops_mod is None:
        raise RuntimeError(f"Não foi possível importar procimg.ops: {_ops_import_err}")

    if hasattr(ops_mod, "dispatch"):
        def _call1(op: str, img: np.ndarray, params: dict[str, Any]) -> np.ndarray:
            return ops_mod.dispatch(op, img, **params)  
        return _call1

    if hasattr(ops_mod, "OPS"):
        def _call2(op: str, img: np.ndarray, params: dict[str, Any]) -> np.ndarray:
            OPS = getattr(ops_mod, "OPS")
            if op not in OPS:
                raise KeyError(f"Operação não encontrada em OPS: {op}")
            fn = OPS[op]

            args = types.SimpleNamespace(**params)
            try:
                return fn(img, args)  
            except TypeError:
                return fn(img, **params)
        return _call2

    def _call3(op: str, img: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        if not hasattr(ops_mod, op):
            raise KeyError(f"Operação não encontrada: {op}")
        fn = getattr(ops_mod, op)
        if not callable(fn):
            raise TypeError(f"Símbolo não chamável em ops: {op}")
        return fn(img, **params) 
    return _call3

def _discover_ops() -> list[str]:
    if ops_mod is None:
        return []
    names: list[str] = []
    if hasattr(ops_mod, "OPS"):
        try:
            names.extend(list(getattr(ops_mod, "OPS").keys()))
        except Exception:
            pass
    for n in dir(ops_mod):
        if n.startswith("_"):
            continue
        obj = getattr(ops_mod, n)
        if callable(obj):
            names.append(n)
    return sorted(set(names))

@app.command("ops")
def list_ops():
    ops = _discover_ops()
    if not ops:
        if _ops_import_err:
            print(f"[red]Falha ao importar procimg.ops:[/red] {_ops_import_err}")
        else:
            print("[yellow]Nenhuma operação encontrada.[/yellow]")
        raise typer.Exit(code=1)
    tbl = Table(title="Operações disponíveis", show_lines=False)
    tbl.add_column("#", justify="right", width=3)
    tbl.add_column("op")
    for i, name in enumerate(ops, 1):
        tbl.add_row(str(i), name)
    print(tbl)

@app.command("run")
def run(
    op: str = typer.Option(..., "--op", help="Nome da operação"),
    inp: str = typer.Option(..., "--in", help="Imagem de entrada (pode ser só o nome do arquivo)"),
    out: str | None = typer.Option(None, "--out", help="Arquivo de saída; se omitido, salva em saidas/<nome>__<op>.png"),
    param: list[str] = typer.Option(None, "--param", help="Parâmetros no formato k=v (pode repetir)"),
):
    params = _parse_params(param)

    in_path = Path(inp)
    if not in_path.exists():
        alt = Path("entradas") / inp
        if alt.exists():
            in_path = alt
        else:
            raise FileNotFoundError(f"Imagem não encontrada: {inp} (tente colocar em 'entradas/' ou passe o caminho completo)")

    img = _read(str(in_path))

    dispatch = _resolve_dispatch()
    out_img = dispatch(op, img, params)

    if isinstance(out_img, dict) and isinstance(out_img.get("img"), np.ndarray):
        out_img = out_img["img"]

    if not isinstance(out_img, np.ndarray):
        raise TypeError("A operação não retornou uma imagem (np.ndarray)")


    if out:
        out_path = Path(out)
    else:
        out_dir = Path("saidas")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{in_path.stem}__{op}.png"

    _write(out_img, str(out_path))
    print(f"[green]OK[/green] {op} -> {out_path}")

if __name__ == "__main__":
    app()
