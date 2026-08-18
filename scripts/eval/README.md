# Avaliação quantitativa do world model

Harness que mede se o world model realmente **prediz o próximo estado da bola**,
e não apenas se os componentes (VAE, action encoder) funcionam isoladamente.
Foi construído em resposta ao parecer do ISCMI2026-VA057, que apontou ausência de
avaliação preditiva quantitativa para as hipóteses H1 (dinâmica) e H2 (ação).

## Arquivos

| Arquivo | Papel |
| --- | --- |
| `generate_holdout.py` | Gera episódios **inéditos** com o mesmo motor de física do dataset (`scenario_1.py`), offscreen e sem limite de FPS. |
| `wm_metrics.py` | Detector de bolas com limiar adaptativo, casamento húngaro, CPE, métricas de pixel, métricas latentes, bootstrap. |
| `wm_runner.py` | Carrega episódios e roda o world model em lote (encode → fusão → ConvLSTM → delta → decode), incluindo rollout autoregressivo. |
| `run_world_model_eval.py` | Orquestra seleção de configuração, avaliação e registro no MLflow. |

## Como rodar

```bash
python scripts/eval/generate_holdout.py --episodes 30
```

```bash
python scripts/eval/run_world_model_eval.py --stage all --dataset holdout
```

Para reavaliar uma configuração específica sem refazer a varredura:

```bash
python scripts/eval/run_world_model_eval.py --stage evaluation --dataset holdout --checkpoint scripts/world_model_weights.pth --memory-frames 6 --memory-stride 2
```

Outros estágios, que não recomputam nada:

- `--stage figures` — redesenha as figuras derivadas de `resultados.json`.
- `--stage best` — registra a configuração vencedora no MLflow a partir do
  `resultados.json` já existente.

Resultados: `scripts/eval/outputs/` (JSON + CSV por frame + figuras) e MLflow
(`mlflow ui --backend-store-uri mlruns`), experimentos
`wfm_world_model_selection` e `wfm_world_model_evaluation`. O relatório com a
leitura dos números está em `docs/relatorio_avaliacao_world_model.md`.

`data/holdout/` é ignorado pelo git (a pasta `data/` inteira é), mas é
**reprodutível**: a geração é determinística dada a semente (`--seed 20260804`,
o default) e a física do `pymunk` também é. Rodar o comando de novo produz
exatamente os mesmos 30 episódios.

## Decisões metodológicas que importam

1. **Conjunto de teste inédito.** O manifesto de treino não foi versionado, então
   qualquer número medido nos vídeos de `scripts/dataset/` é potencialmente
   in-sample. `generate_holdout.py` gera episódios novos com renderização
   idêntica (verificado: diferença de 0.0 pixel no frame inicial).

2. **Baselines no mesmo domínio.** A predição do modelo passa pelo decoder do
   VAE, que perde contraste. Comparar com "copiar o frame anterior" em pixels
   crus seria injusto com o modelo. Por isso as baselines de persistência e de
   velocidade constante também partem de `decode(encode(frame))`, e o **teto do
   VAE** (`decode(mu_verdadeiro)`) é reportado como piso de erro alcançável.

3. **Erro em pixels da tela original.** O modelo trabalha em 64×64, mas o CPE é
   convertido para a tela 800×600, que é a unidade física do cenário.

4. **Alinhamento causal ação/frame.** A bola aparece no frame `ceil(t·60)`; a
   ação é atribuída ao frame anterior, que ainda não a contém. Sem isso, prever
   o efeito do clique seria trivial (a bola já estaria na entrada).

5. **Skill score agregado.** `1 − MSE_modelo / MSE_persistência` calculado sobre
   os erros somados, não como média de razões por frame — em frames estáticos a
   razão explode.
