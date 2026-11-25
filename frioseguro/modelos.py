"""Modelos de aprendizado de máquina do FrioSeguro.

1. Risco de excursão (classificação supervisionada): árvore de decisão x floresta aleatória;
   a melhor pelo F1 no conjunto de teste (separado por viagem, sem vazamento) é usada no painel.
2. Detecção de anomalias do sensor (não supervisionada): Isolation Forest.
3. Agrupamento de viagens pelo comportamento térmico (não supervisionado): k-means.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, silhouette_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .features import FEATURES_ANOMALIA, FEATURES_RISCO

LIMIAR_AMARELO = 0.25
LIMIAR_VERMELHO = 0.5


def _metricas(y, prob, limiar=0.5) -> dict:
    pred = (prob >= limiar).astype(int)
    return {
        "precisao": float(precision_score(y, pred, zero_division=0)),
        "revocacao": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
        "matriz_confusao": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def treinar_risco(feat: pd.DataFrame, passo: int = 5, max_linhas: int = 400_000, seed: int = 42):
    """Treina e compara os classificadores. Devolve (melhor_modelo, relatório)."""
    base = feat[feat["alvo"].notna()]
    base = base[base["min_desde_saida"] % passo == 0]           # 1 amostra a cada 5 min (menos redundância)
    if len(base) > max_linhas:
        base = base.sample(max_linhas, random_state=seed)
    X, y, g = base[FEATURES_RISCO], base["alvo"].astype(int), base["viagem_id"]
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(X, y, g))

    candidatos = {
        "arvore_decisao": DecisionTreeClassifier(max_depth=6, min_samples_leaf=50,
                                                 class_weight="balanced", random_state=seed),
        "floresta_aleatoria": RandomForestClassifier(n_estimators=120, max_depth=14, min_samples_leaf=20,
                                                     class_weight="balanced_subsample", n_jobs=-1,
                                                     random_state=seed),
    }
    relatorio = {"amostras_treino": int(len(tr)), "amostras_teste": int(len(te)),
                 "viagens_teste": int(g.iloc[te].nunique()), "taxa_positivos": float(y.mean()),
                 "modelos": {}}
    melhor, melhor_f1 = None, -1.0
    for nome, modelo in candidatos.items():
        modelo.fit(X.iloc[tr], y.iloc[tr])
        prob = modelo.predict_proba(X.iloc[te])[:, 1]
        m = _metricas(y.iloc[te].values, prob)
        m["importancia"] = dict(sorted(zip(FEATURES_RISCO, map(float, modelo.feature_importances_)),
                                       key=lambda kv: -kv[1]))
        relatorio["modelos"][nome] = m
        if m["f1"] > melhor_f1:
            melhor, melhor_f1, relatorio["escolhido"] = modelo, m["f1"], nome
    return melhor, relatorio


def treinar_anomalia(feat: pd.DataFrame, max_linhas: int = 300_000, contaminacao: float = 0.002, seed: int = 42):
    base = feat.sample(min(max_linhas, len(feat)), random_state=seed)
    modelo = IsolationForest(n_estimators=150, contamination=contaminacao, random_state=seed, n_jobs=-1)
    modelo.fit(base[FEATURES_ANOMALIA])
    relatorio = {"contaminacao": contaminacao}
    if "falha_sensor" in feat:
        aval = feat.sample(min(500_000, len(feat)), random_state=seed + 1)
        pred = (modelo.predict(aval[FEATURES_ANOMALIA]) == -1).astype(int)
        relatorio.update({
            "precisao": float(precision_score(aval["falha_sensor"], pred, zero_division=0)),
            "revocacao": float(recall_score(aval["falha_sensor"], pred, zero_division=0)),
            "f1": float(f1_score(aval["falha_sensor"], pred, zero_division=0)),
        })
    return modelo, relatorio


ATRIB_GRUPO = ["temp_media", "temp_max", "amplitude", "pct_fora_faixa", "n_excursoes", "duracao_min"]
NOMES_GRUPO = ["Estável", "Oscilação moderada", "Aquecimento recorrente", "Crítico"]


def agrupar_viagens(resumo: pd.DataFrame, k: int = 4, seed: int = 42):
    """k-means sobre os indicadores de cada viagem; grupos nomeados pela gravidade média."""
    X = resumo[ATRIB_GRUPO].fillna(0).values
    esc = StandardScaler().fit(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(esc.transform(X))
    rot = km.labels_
    amostra = np.random.default_rng(seed).choice(len(X), size=min(5000, len(X)), replace=False)
    sil = float(silhouette_score(esc.transform(X)[amostra], rot[amostra]))
    gravidade = pd.Series(resumo["pct_fora_faixa"].values).groupby(rot).mean().sort_values()
    nomes = {grp: NOMES_GRUPO[min(i, len(NOMES_GRUPO) - 1)] for i, grp in enumerate(gravidade.index)}
    return km, esc, [nomes[r] for r in rot], {"k": k, "silhueta": sil}


def nivel(prob: float | np.ndarray):
    """Semáforo: verde, amarelo ou vermelho."""
    p = np.asarray(prob)
    return np.where(p >= LIMIAR_VERMELHO, "vermelho", np.where(p >= LIMIAR_AMARELO, "amarelo", "verde"))
