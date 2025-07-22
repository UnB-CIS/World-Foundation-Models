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

WFMs que utilizam abordagens autoregressivas aplicam os mesmos princípios dos modelos de linguagem à geração de ambientes simulados. Nesta arquitetura, a simulação de mundo é gerada por meio da previsão do próximo token, onde cada frame de vídeo é convertido em uma sequência de tokens que são processados sequencialmente pelo modelo. O caráter autoregressivo vem justamente da previsão dos próximos tokens com base na sequência de frames já vista.

### Arquitetura do Sistema

A arquitetura das WFMs autoregressivas segue três componentes principais:

1. **Tokenização de Vídeo**:
   - Os vídeos são inicialmente passados por um tokenizador visual, que transforma cada frame em uma sequência de tokens discretos. Esses tokens são representações compactas dos frames

2. **Núcleo Autoregressivo**:
   - O núcleo do modelo é um Transformer decoder, treinado para prever o próximo token com base na sequência anterior (aqui está o caráter autoregressivo). Para lidar com a estrutura tridimensional dos vídeos (tempo, altura e largura), são utilizados embeddings posicionais espaciais e temporais. Ele também pode receber informações adicionais, como instruções em linguagem natural, por meio de mecanismos de atenção cruzada.

3. **Decodificação**:
   - A geração acontece de forma sequencial, token por token, até que um novo frame seja reconstruído. Há a possibilidade dos tokens gerados serem passados por um decoder de difusão para melhorar a qualidade visual

### Vantagens da Abordagem

Entre os principais pontos positivos dessa arquitetura está sua **escalabilidade**: por herdar a estrutura dos grandes modelos de linguagem (LLMs), ela se adapta bem ao uso de grandes volumes de dados. Outro aspecto importante é a **flexibilidade**: o modelo pode lidar com diferentes tipos de entrada (texto, vídeo, imagem), gerar sequências de comprimentos variados e ser controlado de maneira precisa por prompts

### Limitações

Apesar das vantagens, há desafios inerentes à abordagem. A **geração sequencial** faz com que o processo seja naturalmente mais lento e custoso do ponto de vista computacional, principalmente em vídeos longos. Além disso, como cada passo depende do anterior, **pequenos erros tendem a se propagar** e se amplificar ao longo da sequência, o que pode comprometer a coerência do vídeo gerado. Por fim, o processo de tokenização agressiva, necessário para reduzir o custo computacional, pode introduzir **objetos inesperados** que afetam a fidelidade da simulação, motivo pelo qual, muitas vezes, é necessário aplicar um pós-processamento com modelos de difusão.
