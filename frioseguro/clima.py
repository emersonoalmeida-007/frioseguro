"""Temperatura externa: climatologia aproximada (padrão) ou dados horários da API pública Open-Meteo."""
from __future__ import annotations

import json
import urllib.request

import numpy as np
import pandas as pd

# Média mensal aproximada para a região Sudeste (°C), jan..dez
MEDIA_MENSAL = np.array([26.5, 26.8, 26.0, 24.0, 21.5, 20.0, 19.5, 21.0, 22.5, 24.0, 25.0, 26.0])
AMPLITUDE_DIARIA = 5.0  # metade da variação entre a madrugada e a tarde


def climatologia(data_hora: pd.Series | pd.DatetimeIndex, deslocamento: float = 0.0) -> np.ndarray:
    """Temperatura externa estimada a partir do mês e da hora do dia (pico às 15h)."""
    dh = pd.DatetimeIndex(data_hora)
    hora = dh.hour.values + dh.minute.values / 60.0
    base = MEDIA_MENSAL[dh.month.values - 1]
    return base + deslocamento + AMPLITUDE_DIARIA * np.sin(2 * np.pi * (hora - 9) / 24)


def open_meteo(lat: float, lon: float, inicio: str, fim: str, timeout: int = 20) -> pd.DataFrame:
    """Busca a temperatura horária histórica (Open-Meteo Archive, sem chave).

    Use coordenadas arredondadas (cidade), nunca o endereço de um cliente.
    Retorna colunas data_hora, temp_externa_c.
    """
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat:.2f}&longitude={lon:.2f}&start_date={inicio}&end_date={fim}"
        "&hourly=temperature_2m&timezone=America%2FSao_Paulo"
    )
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        dados = json.load(resp)
    return pd.DataFrame({
        "data_hora": pd.to_datetime(dados["hourly"]["time"]),
        "temp_externa_c": dados["hourly"]["temperature_2m"],
    })


def anexar_temp_externa(leituras: pd.DataFrame, clima_horario: pd.DataFrame | None = None) -> pd.DataFrame:
    """Preenche temp_externa_c: usa a série horária se informada (interpolada por minuto), senão a climatologia."""
    df = leituras.copy()
    if clima_horario is not None and len(clima_horario):
        serie = clima_horario.set_index("data_hora")["temp_externa_c"].sort_index()
        idx = serie.index.union(pd.DatetimeIndex(df["data_hora"]))
        serie = serie.reindex(idx).interpolate(method="time").ffill().bfill()
        df["temp_externa_c"] = serie.reindex(pd.DatetimeIndex(df["data_hora"])).values
    elif "temp_externa_c" not in df or df["temp_externa_c"].isna().all():
        df["temp_externa_c"] = climatologia(df["data_hora"])
    return df
