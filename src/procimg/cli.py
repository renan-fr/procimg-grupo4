#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import sys
import cv2 as cv
from .ops import OPS

def parse_args():
    p = argparse.ArgumentParser(prog="procimg", description="CLI do ProcIMG – operações de cor (OpenCV BGR).")
    p.add_argument("--op", required=True, choices=OPS.keys(), help="Operação a executar")
    p.add_argument("--in", dest="in_path", required=True, help="Caminho da imagem de entrada")
    p.add_argument("--out", dest="out_path", required=False, help="Arquivo de saída (png/jpg)")
    p.add_argument("--lut", default="VIRIDIS", help="LUT (VIRIDIS/JET/TURBO/...) p/ mapear-cores")
    p.add_argument("--cor", default="vermelho", help="Cor alvo (vermelho|verde|azul)")
    p.add_argument("--tol", type=int, default=20, help="Tolerância HSV")
    p.add_argument("--ganho", type=float, default=1.2, help="Ganho p/ realce")
    p.add_argument("--fator", type=float, default=0.2, help="Fator p/ dessaturação seletiva")
    p.add_argument("--cor-origem", dest="cor_origem", default="vermelho", help="Cor origem p/ substituição")
    p.add_argument("--cor-destino", dest="cor_destino", default="azul", help="Cor destino p/ substituição")
    p.add_argument("--hue", type=int, default=20, help="Deslocamento de hue (0..179)")
    return p.parse_args()

def main():
    args = parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        print(f"[ERRO] Entrada não encontrada: {in_path}", file=sys.stderr); sys.exit(1)

    img = cv.imread(str(in_path))
    if img is None:
        print("[ERRO] Falha ao ler a imagem (formato/caminho).", file=sys.stderr); sys.exit(2)

    try:
        result = OPS[args.op](img, args)
    except Exception as e:
        print(f"[ERRO] Execução de '{args.op}': {e}", file=sys.stderr); sys.exit(3)

    out_img = result.get("img", None) if isinstance(result, dict) else result
    if out_img is None:
        print("[ERRO] Nenhuma imagem de saída gerada.", file=sys.stderr); sys.exit(4)

    default_out = Path("dados/saidas") / f"{in_path.stem}_{args.op}.png"
    out_path = Path(args.out_path) if args.out_path else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv.imwrite(str(out_path), out_img):
        print("[ERRO] Falha ao salvar a saída.", file=sys.stderr); sys.exit(5)

    print(f"[OK] {args.op} → {out_path}")

if __name__ == "__main__":
    main()
