"""Simulador de viagens refrigeradas, calibrável pelo teste de rota com data logger.

Gera dados FICTÍCIOS em escala (milhões de leituras por minuto) para treinar e testar os modelos.
Modelo térmico simples (lei de resfriamento de Newton) para dois tipos de embalagem:
- bau_refrigerado: baú com equipamento de frio (setpoint 5 °C); alguns veículos têm o frio degradado
  e algumas viagens têm falha do equipamento;
- caixa_termica: caixa com gelo reciclável; o gelo tem "reserva" que se esgota com o calor recebido.
Aberturas de porta e paradas longas aumentam a troca de calor. O sensor tem ruído e falhas pontuais
(picos), usadas para avaliar a detecção de anomalias.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .clima import climatologia

REGIOES = [
    ("Capital Centro", 0.0), ("Capital Sul", 0.0), ("ABC", -0.3), ("Guarulhos", 0.2),
    ("Campinas", 0.5), ("Sorocaba", 0.0), ("Vale do Paraíba", 0.3), ("Baixada Santista", 1.8),
    ("Ribeirão Preto", 2.0), ("Jundiaí", 0.2), ("Piracicaba", 1.0), ("São José do Rio Preto", 2.5),
]


@dataclass
class Parametros:
    """Parâmetros do modelo térmico. Use calibrar() com o teste de rota real para ajustá-los."""
    k_bau: float = 0.006          # troca de calor do baú fechado (1/min)
    k_caixa: float = 0.0035       # troca de calor da caixa térmica fechada (1/min)
    k_porta_bau: float = 0.012    # troca extra com a porta aberta
    k_porta_caixa: float = 0.035
    frio_normal: float = 0.15     # capacidade do equipamento de frio
    frio_degradado: float = 0.035
    setpoint: float = 5.0
    temp_gelo: float = 4.0
    reserva_gelo_media: float = 250.0   # "reserva" de frio do gelo (°C·min)
    ruido_sensor: float = 0.08
    prob_falha_sensor: float = 0.0015
    prob_falha_frio: float = 0.04
    prob_gelo_mal_condicionado: float = 0.03
    prob_porta_longa: float = 0.10


def catalogo(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rotas e veículos fictícios."""
    rng = np.random.default_rng(seed)
    rotas = []
    for i in range(25):
        nome, desloc = REGIOES[i % len(REGIOES)]
        rotas.append({
            "rota": f"R{i + 1:02d}",
            "regiao": nome,
            "deslocamento_clima": desloc,
            "duracao_base_min": int(rng.integers(120, 420)),
            "paradas_base": int(rng.integers(2, 12)),
        })
    veiculos = []
    for i in range(18):
        emb = "bau_refrigerado" if i < 11 else "caixa_termica"
        veiculos.append({
            "veiculo": f"V{i + 1:02d}",
            "embalagem": emb,
            "frio_degradado": emb == "bau_refrigerado" and i in (6, 9),  # V07 e V10: padrão "escondido"
        })
    return pd.DataFrame(rotas), pd.DataFrame(veiculos)


def _agenda(rng, duracao: int, n_paradas: int, p: Parametros):
    """Vetores minuto a minuto: parado (0/1), porta_aberta (0/1) e lista de eventos."""
    parado = np.zeros(duracao, dtype=np.int8)
    porta = np.zeros(duracao, dtype=np.int8)
    eventos = [(0, "saida", 0)]
    if n_paradas:
        inicios = np.sort(rng.choice(np.arange(20, max(duracao - 30, 21)), size=n_paradas, replace=False))
        for ini in inicios:
            perm = int(rng.integers(5, 26))
            aberta = int(rng.integers(1, 9))
            if rng.random() < p.prob_porta_longa:
                aberta = int(rng.integers(15, 30))
                perm = max(perm, aberta + 3)
            fim = min(ini + perm, duracao)
            parado[ini:fim] = 1
            porta[ini + 1:min(ini + 1 + aberta, duracao)] = 1
            eventos.append((int(ini), "parada", perm))
            eventos.append((int(ini + 1), "porta_aberta", aberta))
    for _ in range(int(rng.poisson(1.2))):  # trânsito parado, sem abrir a porta
        ini = int(rng.integers(10, max(duracao - 10, 11)))
        d = int(rng.integers(10, 40))
        parado[ini:min(ini + d, duracao)] = 1
        eventos.append((ini, "transito", d))
    eventos.append((duracao - 1, "chegada", 0))
    return parado, porta, eventos


def simular_viagem(rng, viagem_id: str, rota: dict, veic: dict, inicio: pd.Timestamp, p: Parametros):
    duracao = int(max(60, rota["duracao_base_min"] * rng.normal(1.0, 0.15)))
    n_paradas = int(max(0, rota["paradas_base"] + rng.integers(-2, 3)))
    parado, porta, eventos = _agenda(rng, duracao, n_paradas, p)

    tempos = inicio + pd.to_timedelta(np.arange(duracao), unit="min")
    text = climatologia(tempos, rota["deslocamento_clima"]) + rng.normal(0, 1.5)
    text = text + np.cumsum(rng.normal(0, 0.03, duracao))  # variação lenta do tempo

    temp = np.empty(duracao)
    bau = veic["embalagem"] == "bau_refrigerado"
    gelo_frio = (not bau) and rng.random() < p.prob_gelo_mal_condicionado
    t_gelo = 1.0 if gelo_frio else p.temp_gelo
    t = rng.normal(4.6, 0.6) if not gelo_frio else rng.normal(2.6, 0.4)
    frio = p.frio_degradado if veic["frio_degradado"] else p.frio_normal
    falha_frio_em = int(rng.integers(30, duracao)) if bau and rng.random() < p.prob_falha_frio else None
    reserva = max(40.0, rng.normal(p.reserva_gelo_media, 70) * (1.0 - 0.012 * (text[0] - 20)))

    for i in range(duracao):
        if bau:
            k = p.k_bau + (p.k_porta_bau if porta[i] else 0.0)
            capacidade = 0.0 if (falha_frio_em is not None and i >= falha_frio_em) else frio
            t += k * (text[i] - t) + capacidade * (p.setpoint - t)
        else:
            k = p.k_caixa + (p.k_porta_caixa if porta[i] else 0.0)
            calor = k * (text[i] - t)
            if reserva > 0:
                reserva -= max(calor, 0.0) * 3.0
                t += 0.25 * calor + 0.05 * (t_gelo - t)
            else:
                t += calor
        temp[i] = t

    lida = temp + rng.normal(0, p.ruido_sensor, duracao)
    falha = (rng.random(duracao) < p.prob_falha_sensor).astype(np.int8)
    if falha.any():
        lida[falha == 1] += rng.choice([-1, 1], falha.sum()) * rng.uniform(6, 15, falha.sum())

    leituras = pd.DataFrame({
        "viagem_id": viagem_id,
        "data_hora": tempos,
        "temperatura_c": np.round(lida, 2),
        "temp_externa_c": np.round(text, 2),
        "parado": parado,
        "porta_aberta": porta,
        "falha_sensor": falha,
    })
    ev = pd.DataFrame([
        {"viagem_id": viagem_id, "data_hora": tempos[m], "evento": e, "duracao_min": d} for m, e, d in eventos
    ])
    info = {
        "viagem_id": viagem_id, "rota": rota["rota"], "regiao": rota["regiao"], "veiculo": veic["veiculo"],
        "embalagem": veic["embalagem"], "inicio": tempos[0], "fim": tempos[-1], "duracao_min": duracao,
        "n_paradas": n_paradas, "falha_frio": falha_frio_em is not None, "gelo_mal_condicionado": gelo_frio,
    }
    return leituras, ev, info


def simular(n_viagens: int = 4000, seed: int = 42, p: Parametros | None = None,
            periodo=("2025-01-01", "2026-10-31")):
    """Simula n viagens e devolve (leituras, eventos, viagens)."""
    p = p or Parametros()
    rng = np.random.default_rng(seed)
    rotas, veiculos = catalogo()
    dias = pd.date_range(*periodo, freq="D")
    horas = np.arange(5, 17)
    peso_horas = np.array([3, 5, 6, 6, 5, 4, 3, 3, 3, 3, 2, 1], dtype=float)
    peso_horas /= peso_horas.sum()
    L, E, V = [], [], []
    for n in range(n_viagens):
        rota = rotas.iloc[int(rng.integers(len(rotas)))].to_dict()
        veic = veiculos.iloc[int(rng.integers(len(veiculos)))].to_dict()
        inicio = dias[int(rng.integers(len(dias)))] + pd.Timedelta(
            hours=int(rng.choice(horas, p=peso_horas)), minutes=int(rng.integers(0, 60)))
        l, e, v = simular_viagem(rng, f"VG{n + 1:06d}", rota, veic, inicio, p)
        L.append(l); E.append(e); V.append(v)
    return pd.concat(L, ignore_index=True), pd.concat(E, ignore_index=True), pd.DataFrame(V)


def calibrar(leituras_reais: pd.DataFrame, embalagem: str = "caixa_termica", p: Parametros | None = None) -> Parametros:
    """Ajusta o coeficiente de troca de calor pela taxa de aquecimento observada no teste de rota real.

    Usa os trechos em que a temperatura sobe: dT/dt ≈ k · (T_ext − T).
    """
    p = p or Parametros()
    df = leituras_reais.sort_values("data_hora")
    dt = df["temperatura_c"].diff().values[1:]
    grad = (df["temp_externa_c"] - df["temperatura_c"]).values[:-1]
    ok = (grad > 3) & (dt > 0)
    if ok.sum() >= 10:
        k = float(np.median(dt[ok] / grad[ok]))
        if embalagem == "caixa_termica":
            p.k_caixa = float(np.clip(k, 0.0005, 0.05))
        else:
            p.k_bau = float(np.clip(k, 0.0005, 0.05))
    return p


def salvar(leituras: pd.DataFrame, eventos: pd.DataFrame, viagens: pd.DataFrame, pasta: Path) -> None:
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    leituras.to_parquet(pasta / "leituras.parquet", index=False)
    eventos.to_parquet(pasta / "eventos.parquet", index=False)
    viagens.to_parquet(pasta / "viagens.parquet", index=False)


def parametros_dict(p: Parametros) -> dict:
    return asdict(p)
