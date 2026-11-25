import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frioseguro import features as F  # noqa: E402
from frioseguro import modelos as M  # noqa: E402
from frioseguro import simulador as S  # noqa: E402
from frioseguro.clima import anexar_temp_externa, climatologia  # noqa: E402
from frioseguro.ingestao import (csv_exemplo, ler_csv_logger, ler_eventos, marcar_eventos,  # noqa: E402
                                 resumo_viagem)


@pytest.fixture(scope="module")
def simulado():
    L, E, V = S.simular(120, seed=3)
    return L.merge(V[["viagem_id", "embalagem"]], on="viagem_id"), E, V


def test_simulacao_estrutura(simulado):
    L, E, V = simulado
    assert len(V) == 120
    assert {"viagem_id", "data_hora", "temperatura_c", "temp_externa_c", "falha_sensor"} <= set(L.columns)
    assert L.groupby("viagem_id").size().min() >= 60
    assert set(E["evento"]) >= {"saida", "chegada"}


def test_simulacao_reprodutivel():
    a = S.simular(5, seed=9)[0]
    b = S.simular(5, seed=9)[0]
    pd.testing.assert_frame_equal(a, b)


def test_climatologia_mais_quente_a_tarde_e_no_verao():
    t = climatologia(pd.to_datetime(["2026-01-15 15:00", "2026-01-15 04:00", "2026-07-15 15:00"]))
    assert t[0] > t[1] and t[0] > t[2]


def test_ler_csv_logger_formato_brasileiro_e_anonimiza():
    df = ler_csv_logger(csv_exemplo(n_min=90).encode())
    assert list(df.columns[:2]) == ["data_hora", "temperatura_c"]
    assert "Nº Série" not in df.columns and len(df) == 90
    assert df["temperatura_c"].between(-5, 30).all()


def test_ler_csv_reamostra_para_um_minuto():
    csv = "Date Time,Temp (C)\n2027-03-18 08:00:00,4.0\n2027-03-18 08:05:00,5.0\n"
    df = ler_csv_logger(csv.encode(), dayfirst=False)
    assert len(df) == 6 and abs(df["temperatura_c"].iloc[1] - 4.2) < 1e-9


def test_eventos_marcam_porta_aberta():
    leit = ler_csv_logger(csv_exemplo(n_min=60).encode())
    ev = ler_eventos(b"data_hora;evento;duracao_min\n18/03/2027 08:10;porta aberta;3\n")
    df = marcar_eventos(leit, ev)
    assert df["porta_aberta"].sum() == 3 and df["parado"].sum() == 3


def test_resumo_viagem_conta_excursoes():
    df = pd.DataFrame({"data_hora": pd.date_range("2027-01-01", periods=6, freq="min"),
                       "temperatura_c": [5, 9, 9, 5, 1, 5]})
    r = resumo_viagem(df)
    assert r["min_fora_faixa"] == 3 and r["n_excursoes"] == 2 and not r["conforme"]


def test_alvo_olha_para_o_futuro():
    n = 60
    temp = np.r_[np.full(40, 5.0), np.full(20, 10.0)]
    df = pd.DataFrame({"viagem_id": "X", "data_hora": pd.date_range("2027-01-01", periods=n, freq="min"),
                       "temperatura_c": temp, "temp_externa_c": 25.0})
    f = F.calcular(df, horizonte=30)
    assert f.loc[39, "alvo"] == 1          # 1 min antes da excursão
    assert f.loc[5, "alvo"] == 0           # longe da excursão
    assert np.isnan(f.loc[45, "alvo"])     # já fora da faixa: não se prevê


def test_mediana_remove_pico_de_sensor():
    temp = np.full(30, 5.0); temp[15] = 18.0
    df = pd.DataFrame({"viagem_id": "X", "data_hora": pd.date_range("2027-01-01", periods=30, freq="min"),
                       "temperatura_c": temp, "temp_externa_c": 25.0})
    f = F.calcular(df)
    assert f["fora_faixa"].sum() == 0 and f.loc[15, "residuo_abs"] > 10


def test_modelos_treinam_e_preveem(simulado):
    L, _, _ = simulado
    feat = F.calcular(L)
    modelo, rel = M.treinar_risco(feat)
    assert rel["escolhido"] in rel["modelos"]
    assert 0 <= rel["modelos"][rel["escolhido"]]["f1"] <= 1
    assert modelo.predict_proba(feat[F.FEATURES_RISCO].head(10)).shape == (10, 2)
    _, rel_a = M.treinar_anomalia(feat)
    assert rel_a["revocacao"] > 0.5        # picos inseridos devem ser encontrados


def test_semaforo():
    assert list(M.nivel([0.1, 0.3, 0.9])) == ["verde", "amarelo", "vermelho"]


def test_calibrar_ajusta_coeficiente():
    n = 200
    text = np.full(n, 30.0); t = np.empty(n); t[0] = 5.0
    for i in range(1, n):
        t[i] = t[i - 1] + 0.01 * (text[i - 1] - t[i - 1])
    df = pd.DataFrame({"data_hora": pd.date_range("2027-01-01", periods=n, freq="min"),
                       "temperatura_c": t, "temp_externa_c": text})
    p = S.calibrar(df, "caixa_termica")
    assert abs(p.k_caixa - 0.01) < 1e-3


def test_anexar_temp_externa_usa_serie_horaria():
    leit = pd.DataFrame({"data_hora": pd.date_range("2027-01-01 10:00", periods=61, freq="min"), "temperatura_c": 5.0})
    clima = pd.DataFrame({"data_hora": pd.to_datetime(["2027-01-01 10:00", "2027-01-01 11:00"]),
                          "temp_externa_c": [20.0, 26.0]})
    df = anexar_temp_externa(leit, clima)
    assert df["temp_externa_c"].iloc[30] == pytest.approx(23.0)
