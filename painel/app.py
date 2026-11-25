"""Painel FrioSeguro (Streamlit). Rode a partir da pasta sistema:  streamlit run painel/app.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from frioseguro import FAIXA_MAX, FAIXA_MIN, HORIZONTE_MIN  # noqa: E402
from frioseguro.ingestao import csv_exemplo, ler_csv_logger, ler_eventos, marcar_eventos  # noqa: E402
from frioseguro.modelos import LIMIAR_AMARELO, LIMIAR_VERMELHO  # noqa: E402
from frioseguro.pipeline import ARTEFATOS, analisar_arquivo  # noqa: E402

CORES = {"verde": "#1a7f37", "amarelo": "#b07d00", "vermelho": "#c62828"}
COR_GRUPO = {"Estável": "#1a7f37", "Oscilação moderada": "#4f81bd",
             "Aquecimento recorrente": "#b07d00", "Crítico": "#c62828"}
NOME_EMB = {"bau_refrigerado": "Baú refrigerado", "caixa_termica": "Caixa térmica"}

st.set_page_config(page_title="FrioSeguro – cadeia fria", page_icon="🧊", layout="wide")


@st.cache_data
def carregar():
    viagens = pd.read_parquet(ARTEFATOS / "viagens_resumo.parquet")
    viagens["embalagem_nome"] = viagens["embalagem"].map(NOME_EMB)
    viagens["teve_excursao"] = viagens["n_excursoes"] > 0
    viagens["mes"] = pd.to_datetime(viagens["inicio"]).dt.to_period("M").dt.to_timestamp()
    viagens["hora_saida"] = pd.to_datetime(viagens["inicio"]).dt.hour
    metricas = json.loads((ARTEFATOS / "metricas.json").read_text(encoding="utf-8"))
    return viagens, metricas


@st.cache_data
def leituras_da_viagem(viagem_id: str) -> pd.DataFrame:
    arq = (ARTEFATOS / "leituras_processadas.parquet").as_posix()
    return duckdb.connect().execute(
        f"SELECT * FROM read_parquet('{arq}') WHERE viagem_id = ? ORDER BY data_hora", [viagem_id]).df()


def fmt(n, casas=0):
    return f"{n:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def grafico_curva(df: pd.DataFrame, titulo: str) -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=FAIXA_MIN, y1=FAIXA_MAX, fillcolor="#1a7f37", opacity=0.08, line_width=0,
                  annotation_text="faixa 2–8 °C", annotation_position="top left")
    fig.add_trace(go.Scatter(x=df["data_hora"], y=df["temperatura_c"], name="Temperatura lida",
                             line=dict(color="#9aa5b1", width=1)))
    if "temperatura_filtrada" in df:
        fig.add_trace(go.Scatter(x=df["data_hora"], y=df["temperatura_filtrada"], name="Temperatura (filtrada)",
                                 line=dict(color="#1f4e79", width=2)))
    if "temp_externa_c" in df:
        fig.add_trace(go.Scatter(x=df["data_hora"], y=df["temp_externa_c"], name="Temperatura externa",
                                 line=dict(color="#e07b39", width=1, dash="dot"), yaxis="y"))
    an = df[df.get("anomalia", 0) == 1]
    if len(an):
        fig.add_trace(go.Scatter(x=an["data_hora"], y=an["temperatura_c"], mode="markers",
                                 name="Leitura anômala (sensor)", marker=dict(color="#7b2cbf", size=9, symbol="x")))
    if "porta_aberta" in df and df["porta_aberta"].sum():
        portas = df[df["porta_aberta"] == 1]
        fig.add_trace(go.Scatter(x=portas["data_hora"], y=[FAIXA_MIN - 1] * len(portas), mode="markers",
                                 name="Porta aberta", marker=dict(color="#c62828", size=5, symbol="line-ns-open")))
    fig.update_layout(title=titulo, yaxis_title="°C", height=420, legend=dict(orientation="h", y=-0.2),
                      margin=dict(t=50, b=10))
    return fig


def grafico_risco(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=LIMIAR_AMARELO, fillcolor=CORES["verde"], opacity=0.08, line_width=0)
    fig.add_hrect(y0=LIMIAR_AMARELO, y1=LIMIAR_VERMELHO, fillcolor=CORES["amarelo"], opacity=0.10, line_width=0)
    fig.add_hrect(y0=LIMIAR_VERMELHO, y1=1, fillcolor=CORES["vermelho"], opacity=0.08, line_width=0)
    fig.add_trace(go.Scatter(x=df["data_hora"], y=df["prob_risco"], name="Risco", line=dict(color="#333", width=1.5)))
    fig.update_layout(title=f"Risco de sair da faixa nos próximos {HORIZONTE_MIN} min", yaxis=dict(range=[0, 1.02],
                      tickformat=".0%"), height=260, margin=dict(t=50, b=10), showlegend=False)
    return fig


# ------------------------------------------------------------------ páginas
def pagina_visao_geral(v: pd.DataFrame, m: dict):
    st.header("Visão geral da cadeia fria")
    st.caption("Dados fictícios e simulados para demonstração. Nenhum dado real de empresa é exibido.")
    with st.expander("Filtros", expanded=True):
        c1, c2, c3 = st.columns(3)
        emb = c1.multiselect("Embalagem", sorted(v["embalagem_nome"].unique()), default=sorted(v["embalagem_nome"].unique()))
        reg = c2.multiselect("Região", sorted(v["regiao"].unique()), default=[])
        per = c3.date_input("Período", (v["inicio"].min().date(), v["inicio"].max().date()))
    f = v[v["embalagem_nome"].isin(emb)]
    if reg:
        f = f[f["regiao"].isin(reg)]
    if isinstance(per, tuple) and len(per) == 2:
        f = f[(f["inicio"].dt.date >= per[0]) & (f["inicio"].dt.date <= per[1])]
    if f.empty:
        st.warning("Nenhuma viagem com esses filtros.")
        return

    k = st.columns(5)
    k[0].metric("Viagens", fmt(len(f)))
    k[1].metric("Leituras analisadas", fmt(f["duracao_min"].sum()))
    k[2].metric("Viagens com excursão", f"{fmt(100 * f['teve_excursao'].mean(), 1)}%")
    k[3].metric("Minutos fora da faixa", fmt(f["min_fora_faixa"].sum()))
    k[4].metric("Leituras anômalas do sensor", fmt(f["n_anomalias"].sum()))

    c1, c2 = st.columns(2)
    por_rota = f.groupby(["rota", "regiao"], as_index=False).agg(
        viagens=("viagem_id", "count"), pct_excursao=("teve_excursao", "mean"))
    por_rota["pct_excursao"] *= 100
    por_rota["rótulo"] = por_rota["rota"] + " – " + por_rota["regiao"]
    fig = px.bar(por_rota.sort_values("pct_excursao", ascending=True).tail(12), x="pct_excursao", y="rótulo",
                 orientation="h", title="Rotas com mais viagens com excursão (%)",
                 labels={"pct_excursao": "% de viagens com excursão", "rótulo": ""}, color_discrete_sequence=["#1f4e79"])
    c1.plotly_chart(fig, use_container_width=True)

    por_hora = f.groupby("hora_saida", as_index=False)["teve_excursao"].mean()
    por_hora["teve_excursao"] *= 100
    fig = px.bar(por_hora, x="hora_saida", y="teve_excursao", title="Excursões por horário de saída (%)",
                 labels={"hora_saida": "Hora de saída", "teve_excursao": "% de viagens com excursão"},
                 color_discrete_sequence=["#e07b39"])
    c2.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    por_mes = f.groupby(["mes", "embalagem_nome"], as_index=False)["teve_excursao"].mean()
    por_mes["teve_excursao"] *= 100
    fig = px.line(por_mes, x="mes", y="teve_excursao", color="embalagem_nome", markers=True,
                  title="Excursões por mês (%)", labels={"mes": "", "teve_excursao": "%", "embalagem_nome": "Embalagem"})
    c1.plotly_chart(fig, use_container_width=True)

    por_veic = f.groupby(["veiculo", "embalagem_nome"], as_index=False).agg(
        pct=("teve_excursao", "mean"), viagens=("viagem_id", "count"))
    por_veic["pct"] *= 100
    fig = px.bar(por_veic.sort_values("pct", ascending=False), x="veiculo", y="pct", color="embalagem_nome",
                 title="Excursões por veículo (%)", labels={"veiculo": "Veículo", "pct": "%", "embalagem_nome": "Embalagem"})
    c2.plotly_chart(fig, use_container_width=True)

    st.subheader("Semáforo das rotas")
    sem = por_rota.copy()
    sem["situação"] = pd.cut(sem["pct_excursao"], [-1, 10, 25, 101], labels=["🟢 Em dia", "🟡 Atenção", "🔴 Crítica"])
    st.dataframe(sem[["rota", "regiao", "viagens", "pct_excursao", "situação"]].sort_values("pct_excursao", ascending=False),
                 hide_index=True, use_container_width=True,
                 column_config={"pct_excursao": st.column_config.NumberColumn("% com excursão", format="%.1f"),
                                "regiao": "Região", "rota": "Rota", "viagens": "Viagens"})


def pagina_viagem(v: pd.DataFrame):
    st.header("Detalhe da viagem")
    c1, c2 = st.columns([1, 2])
    so_exc = c1.checkbox("Só viagens com excursão", value=True)
    lista = v[v["teve_excursao"]] if so_exc else v
    lista = lista.sort_values("risco_max", ascending=False)
    rotulos = (lista["viagem_id"] + " · " + lista["rota"] + " · " + lista["veiculo"] + " · "
               + pd.to_datetime(lista["inicio"]).dt.strftime("%d/%m/%Y %H:%M"))
    escolha = c2.selectbox("Viagem", rotulos.tolist())
    if not escolha:
        return
    vid = escolha.split(" · ")[0]
    info = v[v["viagem_id"] == vid].iloc[0]
    df = leituras_da_viagem(vid)
    k = st.columns(5)
    k[0].metric("Embalagem", NOME_EMB[info["embalagem"]])
    k[1].metric("Duração", f"{info['duracao_min']} min")
    k[2].metric("Temp. máxima", f"{fmt(info['temp_max'], 1)} °C")
    k[3].metric("Minutos fora da faixa", fmt(info["min_fora_faixa"]))
    k[4].metric("Grupo", info["grupo"])
    st.plotly_chart(grafico_curva(df, f"{vid} – {info['rota']} ({info['regiao']}) – {info['veiculo']}"), use_container_width=True)
    st.plotly_chart(grafico_risco(df), use_container_width=True)
    primeiro_alerta = df[(df["prob_risco"] >= LIMIAR_VERMELHO) & (df["fora_faixa"] == 0)]
    primeira_exc = df[df["fora_faixa"] == 1]
    if len(primeira_exc) and len(primeiro_alerta):
        antecedencia = (primeira_exc["data_hora"].iloc[0] - primeiro_alerta["data_hora"].iloc[0]).total_seconds() / 60
        if antecedencia > 0:
            st.success(f"O modelo deu alerta vermelho {antecedencia:.0f} min antes da primeira excursão.")
        else:
            st.info("O alerta vermelho veio junto ou depois do início da excursão.")


def pagina_grupos(v: pd.DataFrame, m: dict):
    st.header("Grupos de viagens (k-means)")
    st.write(f"As viagens foram agrupadas pelo comportamento da temperatura (k = {m['grupos']['k']}, "
             f"silhueta = {fmt(m['grupos']['silhueta'], 2)}).")
    amostra = v.sample(min(3000, len(v)), random_state=1)
    fig = px.scatter(amostra, x="temp_media", y="temp_max", color="grupo", color_discrete_map=COR_GRUPO,
                     hover_data=["viagem_id", "rota", "veiculo", "pct_fora_faixa"],
                     labels={"temp_media": "Temperatura média (°C)", "temp_max": "Temperatura máxima (°C)"},
                     title="Temperatura média x máxima por viagem")
    fig.add_hline(y=FAIXA_MAX, line_dash="dash", line_color="#c62828")
    st.plotly_chart(fig, use_container_width=True)
    resumo = v.groupby("grupo").agg(viagens=("viagem_id", "count"), temp_media=("temp_media", "mean"),
                                    temp_max=("temp_max", "mean"), pct_fora=("pct_fora_faixa", "mean"),
                                    caixa_termica=("embalagem", lambda s: 100 * (s == "caixa_termica").mean()))
    st.dataframe(resumo.round(1), use_container_width=True,
                 column_config={"temp_media": "Temp. média (°C)", "temp_max": "Temp. máx. média (°C)",
                                "pct_fora": "% do tempo fora", "caixa_termica": "% caixa térmica"})


def pagina_modelos(m: dict):
    st.header("Modelos de aprendizado de máquina")
    r = m["risco"]
    st.subheader("1. Risco de excursão (classificação)")
    st.write(f"Alvo: a carga, hoje na faixa, sai de 2–8 °C nos próximos {HORIZONTE_MIN} min. "
             f"Treino com {fmt(r['amostras_treino'])} amostras e teste com {fmt(r['amostras_teste'])} "
             f"({fmt(r['viagens_teste'])} viagens separadas, sem vazamento). Modelo escolhido: **{r['escolhido'].replace('_', ' ')}**.")
    tab = pd.DataFrame({n.replace("_", " "): {k: d[k] for k in ("precisao", "revocacao", "f1", "auc_roc")}
                        for n, d in r["modelos"].items()}).T
    st.dataframe(tab.style.format("{:.3f}"), use_container_width=True)
    c1, c2 = st.columns(2)
    esc = r["modelos"][r["escolhido"]]
    imp = pd.Series(esc["importancia"]).sort_values()
    c1.plotly_chart(px.bar(imp, orientation="h", title="Importância dos atributos",
                           labels={"value": "importância", "index": ""}).update_layout(showlegend=False),
                    use_container_width=True)
    mc = esc["matriz_confusao"]
    c2.plotly_chart(px.imshow(mc, text_auto=True, x=["Previsto: sem excursão", "Previsto: excursão"],
                              y=["Real: sem excursão", "Real: excursão"], color_continuous_scale="Blues",
                              title="Matriz de confusão (teste)"), use_container_width=True)
    a = m["anomalia"]
    st.subheader("2. Leituras anômalas do sensor (Isolation Forest)")
    st.write(f"Precisão {fmt(a.get('precisao', 0), 3)} · revocação {fmt(a.get('revocacao', 0), 3)} · F1 {fmt(a.get('f1', 0), 3)} "
             "(comparado com as falhas de sensor inseridas na simulação).")
    st.subheader("3. Grupos de viagens (k-means)")
    st.write(f"k = {m['grupos']['k']}, coeficiente de silhueta = {fmt(m['grupos']['silhueta'], 3)}.")
    st.caption(f"Base: {fmt(m['n_leituras'])} leituras de {fmt(m['n_viagens'])} viagens; "
               f"processamento em {fmt(m['tempo_processamento_s'], 0)} s; gerado em {m['gerado_em']}.")


def pagina_arquivo():
    st.header("Analisar arquivo do data logger")
    st.write("Envie o CSV exportado pelo data logger (Data/Hora e Temperatura). Colunas como número de série "
             "ou cliente são descartadas. Opcional: a planilha de eventos do teste de rota.")
    c1, c2 = st.columns(2)
    arq = c1.file_uploader("CSV do data logger", type=["csv", "txt"])
    eventos = c2.file_uploader("Planilha de eventos (opcional)", type=["csv"])
    emb = st.radio("Embalagem", list(NOME_EMB), format_func=NOME_EMB.get, horizontal=True, index=1)
    st.download_button("Baixar CSV de exemplo", csv_exemplo(), file_name="exemplo_data_logger.csv", mime="text/csv")
    if arq is None:
        return
    try:
        leituras = ler_csv_logger(arq.getvalue())
    except Exception as e:  # noqa: BLE001
        st.error(f"Não consegui ler o arquivo: {e}")
        return
    if eventos is not None:
        leituras = marcar_eventos(leituras, ler_eventos(eventos.getvalue()))
    feat, resumo = analisar_arquivo(leituras, emb)
    k = st.columns(5)
    k[0].metric("Leituras (1/min)", fmt(resumo["duracao_min"]))
    k[1].metric("Temp. mín/máx", f"{fmt(resumo['temp_min'], 1)} / {fmt(resumo['temp_max'], 1)} °C")
    k[2].metric("Minutos fora da faixa", fmt(resumo["min_fora_faixa"]))
    k[3].metric("Excursões", resumo["n_excursoes"])
    k[4].metric("Laudo", "Conforme" if resumo["conforme"] else "Não conforme")
    st.plotly_chart(grafico_curva(feat, "Curva de temperatura do arquivo"), use_container_width=True)
    st.plotly_chart(grafico_risco(feat), use_container_width=True)
    saida = feat[["data_hora", "temperatura_c", "temperatura_filtrada", "temp_externa_c", "fora_faixa",
                  "prob_risco", "nivel", "anomalia"]]
    st.download_button("Baixar resultado (CSV)", saida.to_csv(index=False, sep=";", decimal=","),
                       file_name="frioseguro_resultado.csv", mime="text/csv")


def pagina_sobre():
    st.header("Sobre o FrioSeguro")
    st.markdown(
        "Projeto Integrador em Computação IV (PJI410) – UNIVESP. Análise de dados em escala de temperatura "
        "capturada por data loggers (IoT) no transporte refrigerado de medicamentos, com aprendizado de máquina "
        "e visualização.\n\n"
        "- **Dados:** simulação calibrada pelo teste de rota com data logger + clima (climatologia ou Open-Meteo). "
        "A transportadora parceira aparece com nome fictício e nenhum dado real identificável é publicado.\n"
        "- **Processamento:** Parquet + DuckDB, pandas.\n"
        "- **Modelos:** árvore de decisão e floresta aleatória (risco), Isolation Forest (sensor), k-means (grupos).\n"
        "- **Referência regulatória:** RDC ANVISA nº 430/2020 (boas práticas de distribuição, armazenagem e transporte)."
    )


def main():
    st.sidebar.title("🧊 FrioSeguro")
    st.sidebar.caption("Cadeia fria de medicamentos – PJI410 UNIVESP")
    pagina = st.sidebar.radio("Navegação", ["Visão geral", "Viagem", "Grupos de viagens", "Modelos",
                                            "Analisar arquivo", "Sobre"])
    if not (ARTEFATOS / "metricas.json").exists():
        st.error("Rode antes: python -m frioseguro.cli tudo --viagens 4000")
        return
    v, m = carregar()
    {"Visão geral": lambda: pagina_visao_geral(v, m), "Viagem": lambda: pagina_viagem(v),
     "Grupos de viagens": lambda: pagina_grupos(v, m), "Modelos": lambda: pagina_modelos(m),
     "Analisar arquivo": pagina_arquivo, "Sobre": pagina_sobre}[pagina]()


main()
