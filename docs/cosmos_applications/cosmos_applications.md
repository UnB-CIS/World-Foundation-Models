# Aplicações da plataforma Cosmos

O NVIDIA Cosmos é uma plataforma de World Foundation Models criada com o intuito de acelerar o desenvolvimento de sistemas de IA física. Ela possui 3 tipos de modelos:

1. **Cosmos Predict**, que consegue prever como um vídeo vai continuar a partir dos primeiros quadros, sendo útil para gerar dados

2. **Cosmos Transfer**, que pega vídeos de várias modalidades diferentes isoladas (como profundidade, segmentação, cor, etc) e reconstrói vídeos realistas com base nessas informações técnicas

3. **Cosmos Reason**, que entende e responde perguntas sobre vídeos, combinando texto e imagem para ajudar na interpretação de dados.

## 1. Geração de datasets sintéticos

Nos últimos anos, com o avanço dos modelos de visão computacional e deep learning, surgiu uma forte demanda por grandes volumes de dados para o treinamento dessas redes. Nesse cenário, uma das aplicações mais promissoras do Cosmos é justamente a geração de dados sintéticos, especialmente em contextos onde a coleta de dados reais é custosa, ou inviável. Assim, o Cosmos entra como um extensor artificial de datasets já existentes, suprindo a demanda por dados treináveis mencionada.

![Faixa de pedestres](src/pedestrian_crosswalk.webp)

> Como exemplo, com base em um vídeo curto ou imagem mostrando um pedestre atravessando na faixa, é possível simular diferentes condições climáticas, de iluminação, de horários, com ângulos diferentes, etc. Essa diversidade vem a ser muito útil para o treinamento de veículos autônomos, por exemplo, já que reduz muito o tamanho do dataset de vídeos reais de treinamento.

Nesse sentido, o **Cosmos Predict** se mostra como o modelo mais adequado, já que seu intuito é a própria geração de dados.

Ademais, uma discussão importante a ser feita é sobre a **viabilidade** da aplicação. Para a geração de vídeos, um dos modelos mais simples é o *Cosmos-Predict2-2B-Video2World*, com 2 Bilhões de parâmetros. Esse modelo, embora seja o mais simples disponibilizado pela Nvidia, não roda localmente em notebooks. Talvez com uma GPU mais potente seja possível rodar com limitações, mas de forma geral o custo computacional é mais alto do que um notebook normal suporta. Assim, abre-se a alternativa de usar cloud computing, que também vem com custos financeiros associados.

| Input image | Output video |
|-------------|--------------|
| ![Input Image](src/example_input.jpg) | [Output Video](src/output.mp4) |

> Exemplo de como o *Cosmos-Predict2-2B-Video2World* pode ser usado. Com uma imagem e texto de input, gera-se um vídeo curto. Essa estratégia pode ser simulada para diferentes situações e aplicações, como por exemplo a da faixa de pedestres mencionada anteriormente.

Outro ponto relevante é a **qualidade dos dados sintéticos**. Embora visualmente realistas, esses dados ainda podem conter vieses ou inconsistências que afetam o treinamento dos modelos. Dessa forma, faz-se necessário o uso de práticas de validação dos outputs gerados e comparação com dados reais de referência. Essa verificação pode ser feita com o **Cosmos-Reason**, destacado anteriormente pela sua capacidade interpretativa de vídeos, ou mesmo manualmente, a depender do tamanho do dataset sintético.

## 2. Pré-visualização de cenas em filmes e jogos
