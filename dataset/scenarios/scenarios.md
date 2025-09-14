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

Dado que o objetivo dos cenários é criar um dataset, a simulação foi configurada para gravar um vídeo, desde o início até o momento em que a janela é fechada. O vídeo é salvo automaticamente com o nome `cenario_1` na pasta `videos` em formato .mp4

### 1. Conclusão

A diversidade do dataset pode ser alcaçada ao juntar a interação com o usuário mencionada e a alteração dos objetos da simulação. Dessa forma, esse cenário é uma forma flexível de testar colisões simples, gravidade e perda de energia.

## Cenário 2: Bolas e obstáculos

> Esse cenário foca em testar a interação entre múltiplos objetos e colisões com superfícies mais complexas e dinâmicas, que podem ser criadas pelo próprio usuário.

