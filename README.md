# FrioSeguro: aprendizado de máquina na cadeia fria de medicamentos

Projeto Integrador em Computação IV (PJI410), UNIVESP.

**Tema norteador:** análise de dados em escala, com dados capturados por IoT (data loggers), aprendizado de máquina e uma interface de visualização.

O FrioSeguro faz três coisas com as leituras de temperatura das viagens refrigeradas (2 °C a 8 °C):
- detecta leituras anômalas do sensor;
- estima o risco de a carga sair da faixa nos próximos 30 minutos;
- agrupa as viagens pelo comportamento térmico.

Os resultados aparecem num painel web.

> Todos os dados deste repositório são **fictícios ou simulados**. A transportadora parceira aparece com nome fictício. Os arquivos reais do teste de rota ficam fora do Git (`dados_teste_rota/` no `.gitignore`) e são anonimizados antes do uso.

**Painel publicado:** https://emersonoalmeida-007-frioseguro-painelapp-ehweoo.streamlit.app

## Arquitetura

```
CSV do data logger ─┐
(teste de rota)     ├─> ingestão (anonimiza, 1 leitura/min) ──┐
Simulador calibrado ┘                                          ├─> Parquet + DuckDB ─> atributos ─> modelos ─> painel Streamlit
Clima (climatologia ou Open-Meteo) ────────────────────────────┘
```

| Módulo | Função |
|---|---|
| `frioseguro/simulador.py` | Modelo térmico (lei de Newton) do baú refrigerado e da caixa térmica com gelo. Inclui portas abertas, paradas, falhas de frio e picos de sensor. Gera milhões de leituras. `calibrar()` ajusta os parâmetros com o teste real. |
| `frioseguro/ingestao.py` | Lê CSVs de data logger com `;` ou `,`, vírgula decimal e nomes de coluna variados. Também lê a planilha de eventos e calcula o resumo e o laudo da viagem. |
| `frioseguro/clima.py` | Temperatura externa: climatologia aproximada ou API pública Open-Meteo, só com coordenadas de cidade. |
| `frioseguro/features.py` | Mediana móvel (remove picos), variações em 5 e 15 min, gradiente externo/interno, hora, mês e alvo futuro. |
| `frioseguro/modelos.py` | Árvore de decisão × floresta aleatória (risco), Isolation Forest (sensor), k-means (grupos) e semáforo. |
| `frioseguro/pipeline.py` | Executa tudo e grava os artefatos e as métricas. |
| `painel/app.py` | Painel com as telas: visão geral, viagem, grupos, modelos e análise de arquivo enviado. |

## Como rodar

```powershell
python -m venv $env:USERPROFILE\.venvs\frioseguro
& $env:USERPROFILE\.venvs\frioseguro\Scripts\python.exe -m pip install -r requirements.txt
& $env:USERPROFILE\.venvs\frioseguro\Scripts\python.exe -m frioseguro.cli tudo --viagens 4000   # ~1,1 milhão de leituras
& $env:USERPROFILE\.venvs\frioseguro\Scripts\python.exe -m streamlit run painel/app.py
& $env:USERPROFILE\.venvs\frioseguro\Scripts\python.exe -m pytest -q
```

Depois do teste de rota, calibre o simulador e gere a base de novo:

```powershell
python -m frioseguro.cli calibrar dados_teste_rota/rota_20270318_01.csv --embalagem caixa_termica
python -m frioseguro.cli tudo --viagens 4000
```

## Avaliação dos modelos

- **Risco:** a separação entre treino e teste é feita **por viagem** (`GroupShuffleSplit`), para que leituras da mesma viagem não fiquem nos dois lados. As métricas são precisão, revocação, F1 e AUC-ROC. O modelo escolhido é o de maior F1.
- **Anomalias:** as previsões são comparadas com os picos de sensor inseridos na simulação.
- **Grupos:** coeficiente de silhueta.

As métricas da última execução ficam em `artefatos/metricas.json` e na tela **Modelos** do painel.

## Referências

- BRASIL. ANVISA. Resolução RDC nº 430, de 8 de outubro de 2020 (boas práticas de distribuição, armazenagem e transporte de medicamentos).
- FACELI, K. et al. *Inteligência artificial: uma abordagem de aprendizado de máquina*. 2. ed. LTC, 2021.
