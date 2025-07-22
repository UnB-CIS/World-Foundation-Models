# Modelos Autorregressivos e Aplicação em World Foundation Models (WFMs)

## O que é um Modelo Autorregressivo?

Um modelo autorregressivo (AR) é um tipo de modelo estatístico que prevê valores futuros em uma sequência com base em seus próprios valores passados. O termo "autorregressivo" reflete a ideia de que o modelo "faz regressão sobre si mesmo", ou seja, as previsões são feitas a partir das observações anteriores. Nele, temos a relação:
  
- **Entrada**: Observações passadas (ex: palavras anteriores em uma frase, quadros em um vídeo).

- **Saída**: Previsão do próximo valor na sequência (ex: próxima palavra, próximo quadro).

### Exemplos de uso

Modelos autoregressivos são muito usados em áreas como processamento de linguagem natural (PLN) e séries temporais, devido à sua capacidade de capturar dependências sequenciais e temporais. Alguns exemplos de modelos autoregressivos são:

- **Séries temporais**: usados para prever dados sequenciais, como preços de ações, previsão do tempo ou tráfego de dados

- **PLN**: modelos como o GPT funcionam com abordagem autoregressiva, gerando a próxima palavra com base nas palavras anteriores, sequencialmente

## O que são World Foundation Models (WFMs)?

World Foundation Models (WFMs) são modelos de IA que simulam ou geram ambientes dinâmicos que simulam o mundo real em algum aspecto. Esses modelos são fundamentais para sistemas que possuem um impacto físico no mundo real, como robôs ou veículos autônomos. Para que essas IAs possam operar no mundo físico, elas primeiro precisam de ambientes de treinamento seguros, de forma a aprender como as condições do mundo real atuam na sua área de atuação específica antes de poderem agir de fato no mundo real. Dessa forma, WFMs proporcionam esses ambientes de treinamento, agindo como intermediário importante para o treinamento de modelos de IA com atuação no mundo físico.

Na construção de WFMs, usamos uma abordagem em duas etapas, de pré-treinamento e pós-treinamento, que equilibra a capacidade de generalização com a especialização. Na primeira fase, o modelo é exposto a uma grande variedade de dados de vídeo, absorvendo padrões do mundo real em larga escala. Isso cria uma base capaz de entender diversos contextos. Depois, refinamos esse conhecimento geral com dados específicos de uma área de atuação, como a robótica ou direção autônoma. Assim, o modelo se adapta às nuances do ambiente real de atuação, sem perder sua versatilidade.

## WFM baseado em modelo autoregressivo
