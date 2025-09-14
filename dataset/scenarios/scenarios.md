# Cenários para criação do dataset

## Cenário 1: Bola e chão

> Esse cenário foca em testar colisões, gravidade e perda de energia. Ele foi projetado para ser interativo, proporcionando uma variedade de possibilidades de simulação.

### 1. Objetivos

- Verificar o comportamento de objetos em queda livre.

- Testar a colisão com uma superfície estática (chão) e com outros objetos (bolas dinâmicamente inseridas)

- Analisar o quique e a perda de energia (elasticidade).

### 1. Objetos

1. **Chão (Objeto estático)**: O chão é a única superfície estática do cenário, agindo como a principal barreira de colisão. Sua elasticidade e fricção podem ser alteradas em código para simular diferentes superfícies.

2. **Bolas (Objetos dinâmicos)**: As bolas são os objetos dinâmicos do cenário, e podem ser criadas com um clique do mouse esquerdo. Sua massa, raio, elasticidade e fricção podem ser alterados em código. Além de colidirem com o chão, as bolas podem colidir entre si, o que permite a análise de colisões entre corpos dinâmicos e reações em cadeia.

### 1. Interação com o usuário

A principal forma de interação com o usuário é por meio da adição de bolas usando o mouse. Essa interação permite fazer várias aplicações do mesmo cenário, contribuindo para a geração de um dataset mais diversificado. Clicando com o botão esquerdo do mouse, será adicionada uma nova bola na posição que o mouse se encontra.

### 1. Saída em forma de vídeo

Dado que o objetivo dos cenários é criar um dataset, a simulação foi configurada para gravar um vídeo, desde o início até o momento em que a janela é fechada. O vídeo é salvo automaticamente com o nome `cenario1: data`, em que a "data" se refere ao dia e horário que a simulação foi salva. A simulação é salva na pasta `videos` em formato .mp4

### 1. Conclusão

A diversidade do dataset pode ser alcaçada ao juntar a interação com o usuário mencionada e a alteração dos objetos da simulação. Dessa forma, esse cenário é uma forma flexível de testar colisões simples, gravidade e perda de energia.

## Cenário 2: Bolas e obstáculos

> Esse cenário foca em testar a interação entre múltiplos objetos e colisões com superfícies mais complexas e dinâmicas, que podem ser criadas pelo próprio usuário.

### 2. Objetivos

- Testar a física de colisão em superfícies não-planas (como rampas e curvas) 

- Analisar a interação entre diferentes formas de objetos (círculos e quadrados).

- Avaliar a dinâmica de empilhamento e dispersão de um grande número de objetos.

### 2. Objetos

1. Obstáculos e Superfícies (Objetos Estáticos): Dessa vez, as superfícies estáticas são definidas diretamente pelo usuário, e atuam como paredes e obstáculos, com propriedades de elasticidade e fricção ajustáveis.

2. Bolas e quadrados (Objetos Dinâmicos): O cenário agora suporta dois tipos de formas dinâmicas, que podem ser alternadas pelo usuário. A adição dos quadrados proporciona uma interação mais complexa, podendo deslizar ou tombar ao interagir com os obstáculos.

### 2. Interação com o usuário

A interação com o usuário é mais presente nesse cenário. Nele, podemos realizar:

- Desenho Livre: Clique e arraste o botão direito do mouse para desenhar superfícies estáticas livres. Ao soltar o botão, o desenho é finalizado e se torna uma barreira de colisão.

- Criar objetos: pressione `b` para selecionar o modo "bolas" ou `q` para selecionar o modo "quadrados". Apertar o botão esquerdo do mouse cria um objeto selecionado novo.

- Criação em massa: Segurar o botão `Shift` enquanto aperta o botão esquerdo do mouse cria múltiplos objetos simultaneamente.

### 2. Saída em forma de vídeo

Novamente, a saída é salva em formato de vídeo com o dia e horario da simulação, e fica armazenado na pasta `videos` sob o formato .mp4

### 2. Conclusão

A adição de obstáculos dinâmicos determinados pelo usuário, a funcionalidade de criação em massa e a possibilidade de simular mais uma forma geométrica contribuem para a flexibilidade do cenário, o que ajudará na criação do dataset futuramente.
