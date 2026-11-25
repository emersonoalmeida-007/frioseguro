"""Leitura dos arquivos CSV exportados pelos data loggers e da planilha de eventos do teste de rota.

Aceita separador ; ou , , vírgula decimal e nomes de coluna comuns (Data/Hora, Date Time,
Temperatura, Temp (°C), Umidade...). Mantém apenas data, hora, temperatura e umidade:
qualquer outra coluna (número de série, cliente, placa...) é descartada (anonimização).
"""
from __future__ import annotations

import io
import re
import unicodedata

import numpy as np
import pandas as pd


def _normaliza(nome: str) -> str:
    nome = unicodedata.normalize("NFKD", str(nome)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", nome).strip("_")


def _acha(colunas: list[str], padroes: list[str]) -> str | None:
    for c in colunas:
        n = _normaliza(c)
        if any(re.search(p, n) for p in padroes):
            return c
    return None


def _numero(serie: pd.Series) -> pd.Series:
    if serie.dtype.kind in "if":
        return serie.astype(float)
    s = serie.astype(str).str.strip().str.replace(r"[^0-9,.\-]", "", regex=True)
    tem_virgula = s.str.contains(",", regex=False)
    s = s.where(~tem_virgula, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(s, errors="coerce")


def ler_csv_logger(arquivo, dayfirst: bool = True) -> pd.DataFrame:
    """Lê um CSV de data logger e devolve data_hora, temperatura_c (e umidade_pct, se houver), por minuto."""
    if isinstance(arquivo, (bytes, bytearray)):
        arquivo = io.BytesIO(arquivo)
    bruto = pd.read_csv(arquivo, sep=None, engine="python", encoding_errors="replace", dtype=str)
    cols = list(bruto.columns)

    c_dh = _acha(cols, [r"data_?hora", r"date_?time", r"timestamp", r"^horario$"])
    c_data = _acha(cols, [r"^data$", r"^date$"])
    c_hora = _acha(cols, [r"^hora$", r"^time$"])
    c_temp = _acha(cols, [r"temp"])
    c_umid = _acha(cols, [r"umid", r"humid", r"^rh"])
    if c_temp is None or (c_dh is None and c_data is None):
        raise ValueError("Não encontrei as colunas de data/hora e temperatura no arquivo.")

    if c_dh is not None:
        dh = pd.to_datetime(bruto[c_dh], dayfirst=dayfirst, errors="coerce")
    else:
        texto = bruto[c_data].astype(str) + " " + (bruto[c_hora].astype(str) if c_hora else "")
        dh = pd.to_datetime(texto, dayfirst=dayfirst, errors="coerce")

    df = pd.DataFrame({"data_hora": dh, "temperatura_c": _numero(bruto[c_temp])})
    if c_umid is not None:
        df["umidade_pct"] = _numero(bruto[c_umid])
    df = df.dropna(subset=["data_hora", "temperatura_c"]).sort_values("data_hora")
    df = df.drop_duplicates("data_hora")
    return reamostrar(df)


def reamostrar(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """Coloca as leituras em grade de 1 minuto (interpolação linear, sem extrapolar)."""
    s = df.set_index("data_hora").sort_index()
    s = s.resample(freq).mean().interpolate(method="time", limit_area="inside")
    return s.dropna(subset=["temperatura_c"]).reset_index()


def ler_eventos(arquivo, dayfirst: bool = True) -> pd.DataFrame:
    """Planilha de eventos do teste: data_hora; evento; duração (min); observação."""
    if isinstance(arquivo, (bytes, bytearray)):
        arquivo = io.BytesIO(arquivo)
    bruto = pd.read_csv(arquivo, sep=None, engine="python", dtype=str)
    cols = list(bruto.columns)
    c_dh = _acha(cols, [r"data_?hora", r"date_?time"])
    c_ev = _acha(cols, [r"evento", r"event"])
    c_du = _acha(cols, [r"durac", r"minut"])
    ev = pd.DataFrame({
        "data_hora": pd.to_datetime(bruto[c_dh], dayfirst=dayfirst, errors="coerce"),
        "evento": bruto[c_ev].str.strip().str.lower(),
        "duracao_min": _numero(bruto[c_du]).fillna(0) if c_du else 0,
    })
    return ev.dropna(subset=["data_hora"])


def marcar_eventos(leituras: pd.DataFrame, eventos: pd.DataFrame) -> pd.DataFrame:
    """Cria as colunas parado e porta_aberta (0/1) a partir da planilha de eventos."""
    df = leituras.copy()
    df["parado"] = 0
    df["porta_aberta"] = 0
    for _, e in eventos.iterrows():
        fim = e["data_hora"] + pd.Timedelta(minutes=float(e["duracao_min"] or 0))
        m = (df["data_hora"] >= e["data_hora"]) & (df["data_hora"] < fim)
        if "porta" in e["evento"]:
            df.loc[m, "porta_aberta"] = 1
            df.loc[m, "parado"] = 1
        elif any(p in e["evento"] for p in ("parada", "entrega", "transito", "trânsito")):
            df.loc[m, "parado"] = 1
    return df


def resumo_viagem(df: pd.DataFrame, faixa=(2.0, 8.0)) -> dict:
    """Indicadores de uma viagem (usa a temperatura já sem picos de sensor, se a coluna existir)."""
    col = "temperatura_filtrada" if "temperatura_filtrada" in df else "temperatura_c"
    t = df[col]
    fora = (t < faixa[0]) | (t > faixa[1])
    blocos = (fora != fora.shift()).cumsum()[fora]
    return {
        "inicio": df["data_hora"].min(), "fim": df["data_hora"].max(),
        "duracao_min": int(len(df)),
        "temp_min": float(t.min()), "temp_max": float(t.max()), "temp_media": float(t.mean()),
        "min_fora_faixa": int(fora.sum()),
        "n_excursoes": int(blocos.nunique()) if len(blocos) else 0,
        "pct_fora_faixa": float(100 * fora.mean()) if len(df) else 0.0,
        "conforme": bool(fora.sum() == 0),
    }


def csv_exemplo(n_min: int = 240, inicio: str = "2027-03-18 08:00", seed: int = 1) -> str:
    """Gera um CSV no formato típico de data logger (para testes e demonstração)."""
    rng = np.random.default_rng(seed)
    t = 4.5 + np.cumsum(rng.normal(0.012, 0.05, n_min))
    dh = pd.date_range(inicio, periods=n_min, freq="1min")
    linhas = ["Nº Série;Data/Hora;Temperatura (°C);Umidade (%)"]
    for d, v in zip(dh, t):
        linhas.append(f"SN-000123;{d:%d/%m/%Y %H:%M:%S};{v:.1f}".replace(".", ",") + ";55")
    return "\n".join(linhas)
