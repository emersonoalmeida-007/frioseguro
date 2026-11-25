"""Engenharia de atributos por minuto, a partir apenas do que um data logger e o clima fornecem."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import FAIXA_MAX, FAIXA_MIN, HORIZONTE_MIN

FEATURES_RISCO = [
    "temperatura_filtrada", "d5", "d15", "media15", "desvio15",
    "temp_externa_c", "gradiente", "hora", "mes", "min_desde_saida", "embalagem_caixa",
]
FEATURES_ANOMALIA = ["residuo", "d1_abs", "residuo_abs"]


def _por_viagem(df: pd.DataFrame, col: str, func):
    return df.groupby("viagem_id", sort=False)[col].transform(func)


def calcular(df: pd.DataFrame, horizonte: int = HORIZONTE_MIN) -> pd.DataFrame:
    """Recebe leituras (viagem_id, data_hora, temperatura_c, temp_externa_c[, embalagem]) e devolve atributos.

    - temperatura_filtrada: mediana móvel de 5 min (remove picos isolados do sensor);
    - d5/d15: variação em 5 e 15 min; media15/desvio15: média e desvio em 15 min;
    - gradiente: temperatura externa menos interna (força que aquece a carga);
    - alvo: 1 se a carga, hoje dentro da faixa, sair de 2–8 °C nos próximos `horizonte` minutos.
    """
    df = df.sort_values(["viagem_id", "data_hora"]).reset_index(drop=True)
    df["temperatura_filtrada"] = _por_viagem(
        df, "temperatura_c", lambda s: s.rolling(5, center=True, min_periods=1).median())
    df["residuo"] = df["temperatura_c"] - df["temperatura_filtrada"]
    df["residuo_abs"] = df["residuo"].abs()
    df["d1_abs"] = _por_viagem(df, "temperatura_c", lambda s: s.diff().abs()).fillna(0)
    df["d5"] = _por_viagem(df, "temperatura_filtrada", lambda s: s.diff(5)).fillna(0)
    df["d15"] = _por_viagem(df, "temperatura_filtrada", lambda s: s.diff(15)).fillna(0)
    df["media15"] = _por_viagem(df, "temperatura_filtrada", lambda s: s.rolling(15, min_periods=1).mean())
    df["desvio15"] = _por_viagem(df, "temperatura_filtrada", lambda s: s.rolling(15, min_periods=2).std()).fillna(0)
    df["gradiente"] = df["temp_externa_c"] - df["temperatura_filtrada"]
    dh = pd.DatetimeIndex(df["data_hora"])
    df["hora"] = dh.hour + dh.minute / 60.0
    df["mes"] = dh.month
    df["min_desde_saida"] = df.groupby("viagem_id", sort=False).cumcount()
    emb = df["embalagem"] if "embalagem" in df else pd.Series("caixa_termica", index=df.index)
    df["embalagem_caixa"] = (emb == "caixa_termica").astype(np.int8)

    fora = ((df["temperatura_filtrada"] < FAIXA_MIN) | (df["temperatura_filtrada"] > FAIXA_MAX)).astype(np.int8)
    df["fora_faixa"] = fora
    futuro = df.assign(_f=fora).groupby("viagem_id", sort=False)["_f"].transform(
        lambda s: s[::-1].rolling(horizonte, min_periods=1).max()[::-1].shift(-1))
    df["alvo"] = futuro.where(fora == 0)  # só faz sentido prever para quem ainda está dentro da faixa
    return df
