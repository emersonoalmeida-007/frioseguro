"""Linha de comando.

    python -m frioseguro.cli simular --viagens 4000      # gera dados fictícios em dados/
    python -m frioseguro.cli calibrar dados_teste_rota/rota_20270318_01.csv
    python -m frioseguro.cli treinar                       # atributos, modelos e artefatos do painel
    python -m frioseguro.cli tudo --viagens 4000
"""
from __future__ import annotations

import argparse
import json
import time

from . import simulador
from .clima import anexar_temp_externa
from .ingestao import ler_csv_logger
from .pipeline import DADOS, RAIZ, processar


def main(argv=None):
    ap = argparse.ArgumentParser(prog="frioseguro")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simular"); s.add_argument("--viagens", type=int, default=4000); s.add_argument("--seed", type=int, default=42)
    c = sub.add_parser("calibrar"); c.add_argument("csv"); c.add_argument("--embalagem", default="caixa_termica")
    sub.add_parser("treinar")
    t = sub.add_parser("tudo"); t.add_argument("--viagens", type=int, default=4000); t.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    param_arq = RAIZ / "artefatos" / "parametros_calibrados.json"
    if a.cmd in ("simular", "tudo"):
        p = simulador.Parametros()
        if param_arq.exists():
            p = simulador.Parametros(**json.loads(param_arq.read_text(encoding="utf-8")))
            print("usando parâmetros calibrados pelo teste de rota")
        t0 = time.time()
        L, E, V = simulador.simular(a.viagens, a.seed, p)
        simulador.salvar(L, E, V, DADOS)
        print(f"{len(V)} viagens, {len(L):,} leituras simuladas em {time.time() - t0:.0f}s".replace(",", "."))
    if a.cmd == "calibrar":
        leit = anexar_temp_externa(ler_csv_logger(a.csv))
        p = simulador.calibrar(leit, a.embalagem)
        param_arq.parent.mkdir(exist_ok=True)
        param_arq.write_text(json.dumps(simulador.parametros_dict(p), indent=2), encoding="utf-8")
        print("parâmetros calibrados:", param_arq)
    if a.cmd in ("treinar", "tudo"):
        processar()


if __name__ == "__main__":
    main()
