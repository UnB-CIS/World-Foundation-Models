# Documentação — Avaliação da Biblioteca de Física 2D

## 1. Como cada simulação é gerada

A biblioteca permite configurar um **mundo físico 2D** com corpos dinâmicos (que se movem e sofrem ação da gravidade) e corpos estáticos (que servem como chão ou barreiras).  

- **Queda de objetos**  
  Criada a partir de múltiplos corpos dinâmicos empilhados acima de uma superfície estática.  
  A gravidade faz com que eles caiam e interajam entre si, reproduzindo colisões e empilhamento.  

- **Colisão de objetos**  
  Gerada ao adicionar um corpo extra (como uma “bala” ou “esfera”) com velocidade ou impulso direcionado contra outros corpos já existentes.  
  A biblioteca calcula automaticamente as interações e a resposta física (deslocamento, rotação, queda etc.).  

Esses dois cenários básicos mostram que a biblioteca consegue lidar tanto com movimentações naturais (queda) quanto com interações externas (impacto).

## 2. Tipo de saída do vídeo (tamanho e resolução)

- Cada frame da simulação pode ser capturado diretamente como imagem (`.png`).  
- A resolução depende do **tamanho da janela de renderização** configurada, podendo variar (exemplo comum: `800x600`).  
- As imagens podem ser unidas em um vídeo (`.mp4`) usando ferramentas como `ffmpeg`.  
- A taxa de quadros (FPS) é configurável no agendamento do loop da simulação (60 FPS é o padrão).  

Portanto, é possível documentar ou compartilhar os testes tanto em forma de imagens quanto em vídeo de alta qualidade.

## 3. Facilidade de alterações

- **Parâmetros físicos** (gravidade, massa, atrito, restituição, impulso inicial) são facilmente ajustáveis.  
- **Cenários de teste** podem ser montados apenas alterando listas de objetos adicionados ao mundo (sem precisar modificar a lógica principal).  
- **Renderização e saída** também são flexíveis: é possível salvar dados em CSV/JSON para análise numérica, além de capturar imagens e vídeos.  

Isso mostra que a biblioteca tem boa modularidade e permite implementar novos testes sem esforço significativo.

## Conclusão

Os testes de **queda de objetos** e **colisão** demonstram que a biblioteca é suficiente para representar interações físicas básicas em 2D.  

Ela fornece tanto as ferramentas para simulação realista quanto para captura e documentação dos resultados. Alterar os parâmetros ou criar novos cenários é direto, o que facilita validar diferentes hipóteses dentro do nosso problema.
