"""Pipeline completo: dados brutos (Parquet) -> atributos -> modelos -> resultados para o painel."""
from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from . import features as F
from . import modelos as M
from .clima import anexar_temp_externa
from .ingestao import resumo_viagem

RAIZ = Path(__file__).resolve().parents[1]
DADOS = RAIZ / "dados"
ARTEFATOS = RAIZ / "artefatos"

COLUNAS_PAINEL = ["viagem_id", "data_hora", "temperatura_c", "temperatura_filtrada", "temp_externa_c",
                  "parado", "porta_aberta", "fora_faixa", "prob_risco", "anomalia"]


def carregar_brutos(pasta: Path = DADOS) -> pd.DataFrame:
    """Une leituras e atributos da viagem com DuckDB (consulta direta nos arquivos Parquet)."""
    con = duckdb.connect()
    return con.execute(f"""
        SELECT l.*, v.embalagem
        FROM read_parquet('{(pasta / "leituras.parquet").as_posix()}') l
        JOIN read_parquet('{(pasta / "viagens.parquet").as_posix()}') v USING (viagem_id)
    """).df()


def resumo_por_viagem(feat: pd.DataFrame) -> pd.DataFrame:
    g = feat.groupby("viagem_id", sort=False)
    t = g["temperatura_filtrada"]
    fora = feat["fora_faixa"].astype(bool)
    inicio_exc = fora & ~fora.groupby(feat["viagem_id"]).shift(fill_value=False)
    r = pd.DataFrame({
        "temp_media": t.mean(), "temp_max": t.max(), "temp_min": t.min(),
        "amplitude": t.max() - t.min(),
        "min_fora_faixa": g["fora_faixa"].sum(),
        "pct_fora_faixa": 100 * g["fora_faixa"].mean(),
        "n_excursoes": inicio_exc.groupby(feat["viagem_id"]).sum(),
        "duracao_min": g.size(),
        "risco_max": g["prob_risco"].max(),
        "n_anomalias": g["anomalia"].sum(),
    }).reset_index()
    return r


def processar(pasta: Path = DADOS, saida: Path = ARTEFATOS, verbose: bool = True) -> dict:
    t0 = time.time()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    saida.mkdir(parents=True, exist_ok=True)

    brutos = carregar_brutos(pasta)
    log(f"leituras carregadas: {len(brutos):,}".replace(",", "."))
    feat = F.calcular(brutos)
    log(f"atributos calculados ({time.time() - t0:.0f}s)")

    modelo_risco, rel_risco = M.treinar_risco(feat)
    log(f"risco: escolhido {rel_risco['escolhido']} F1={rel_risco['modelos'][rel_risco['escolhido']]['f1']:.3f}")
    modelo_anom, rel_anom = M.treinar_anomalia(feat)
    log(f"anomalia: F1={rel_anom.get('f1', float('nan')):.3f}")

    feat["prob_risco"] = modelo_risco.predict_proba(feat[F.FEATURES_RISCO])[:, 1]
    feat.loc[feat["fora_faixa"] == 1, "prob_risco"] = 1.0
    feat["anomalia"] = (modelo_anom.predict(feat[F.FEATURES_ANOMALIA]) == -1).astype(np.int8)

    resumo = resumo_por_viagem(feat)
    km, esc, nomes, rel_grupo = M.agrupar_viagens(resumo)
    resumo["grupo"] = nomes
    viagens = pd.read_parquet(pasta / "viagens.parquet").merge(resumo, on=["viagem_id", "duracao_min"], how="left")
    log(f"agrupamento: silhueta={rel_grupo['silhueta']:.3f}")

    feat[COLUNAS_PAINEL].to_parquet(saida / "leituras_processadas.parquet", index=False)
    viagens.to_parquet(saida / "viagens_resumo.parquet", index=False)
    joblib.dump({"modelo": modelo_risco, "features": F.FEATURES_RISCO}, saida / "modelo_risco.joblib")
    joblib.dump({"modelo": modelo_anom, "features": F.FEATURES_ANOMALIA}, saida / "modelo_anomalia.joblib")
    joblib.dump({"kmeans": km, "escala": esc}, saida / "modelo_grupos.joblib")
    metricas = {
        "gerado_em": pd.Timestamp.now().isoformat(timespec="seconds"),
        "n_leituras": int(len(feat)), "n_viagens": int(feat["viagem_id"].nunique()),
        "risco": rel_risco, "anomalia": rel_anom, "grupos": rel_grupo,
        "tempo_processamento_s": round(time.time() - t0, 1),
    }
    (saida / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"concluído em {metricas['tempo_processamento_s']}s")
    return metricas


def analisar_arquivo(leituras: pd.DataFrame, embalagem: str = "caixa_termica",
                     clima_horario: pd.DataFrame | None = None, artefatos: Path = ARTEFATOS):
    """Aplica os modelos treinados a um arquivo novo de data logger (já lido por ingestao.ler_csv_logger)."""
    df = anexar_temp_externa(leituras, clima_horario)
    df["viagem_id"] = "ARQUIVO"
    df["embalagem"] = embalagem
    feat = F.calcular(df)
    risco = joblib.load(artefatos / "modelo_risco.joblib")
    anom = joblib.load(artefatos / "modelo_anomalia.joblib")
    feat["prob_risco"] = risco["modelo"].predict_proba(feat[risco["features"]])[:, 1]
    feat.loc[feat["fora_faixa"] == 1, "prob_risco"] = 1.0
    feat["anomalia"] = (anom["modelo"].predict(feat[anom["features"]]) == -1).astype(np.int8)
    feat["nivel"] = M.nivel(feat["prob_risco"])
    return feat, resumo_viagem(feat)
