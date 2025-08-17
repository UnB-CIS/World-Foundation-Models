# Escopo - Projeto World Foundation Model ()

## Visão Geral do projeto

Desenvolver, via simulações físicas 2D, um World Foundation Model capaz de aprender dinâmicas e cinemática de objetos a partir de observações visuais (vídeo/imagens) e prever/planejar evoluções futuras do sistema. O usuário descreve a cena e a tarefa por prompt; o sistema instancia a simulação, executa/prevê rollouts e retorna vídeo, métricas e ações planejadas.

## Problema

Falta um modelo unificado, orientado por dados, que:
*Generalize para novas combinações de objetos/propriedades (massa, atrito, elasticidade) em 2D.
*Preveja estados futuros com consistência física por dezenas de passos.
\*Planeje ações para atingir metas (p.ex., “fazer o bloco vermelho tocar o alvo”) sem ajustar hiperparâmetros por tarefa.

## Objetivos

- Criar ambiente 2D para a simulação do prompt do usuário.
- Permitir a interação do usuário com o ambiente através de prompts fornecidos ao sistema.
- Fornecer a representação do sistema proposto pelo usuário através de vídeos.

## Fora do Escopo

- 3D, visão estéreo ou robótica real.

- Aprendizado por reforço end-to-end em hardware físico.

- Contato deformável/fluídos complexos; multiagente competitivo.

- RLHF/feedback humano avançado; integração mobile.

## Usuarios e casos de uso

## Abordagem Ténica

Este projeto

## Métricas de Sucesso

## Entregáveis

- FrontEnd que permita a interação do usuário com o sistema para captar o seu prompt.
- Vídeo que representa o resultado final da simulação praticado pelo usuário.

## Riscos & Suposições

Poder computacional falho, dificuldades de manejar cargas de atividades da equipe.

## Links úteis

## Referências | References

[Especificação do projeto](https://docs.google.com/document/d/1GqxDtGbsp0xNqUrYcW_h2VUKyDDuHM6sbOMI6WNnzyQ/edit?tab=t.0)
