# Escopo - Projeto World Foundation Model ()

## Visão Geral do projeto

Desenvolver, via simulações físicas 2D, um World Foundation Model capaz de aprender dinâmicas e cinemática de objetos a partir de observações visuais (vídeo/imagens) e prever/planejar evoluções futuras do sistema. O usuário descreve a cena e a tarefa por prompt; o sistema instancia a simulação, executa/prevê rollouts e retorna vídeo, métricas e ações planejadas.

## Problema

- Membros não possuem experiência prática com arquitetura de _world models_, de tal maneira que o projeto se faz necessário para permitir que os membros se capacitem e compreendam a profundidade tanto dos fundamentos quanto da implementação de modelos generativos aplicado à simulação física de comportamento de objetos diante de diferentes cenários dados por meio de prompts do usuário.
  Falta um modelo unificado, orientado por dados, que:
  **Generalize** para novas combinações de objetos/propriedades (massa, atrito, elasticidade) em 2D.
  **Preveja** estados futuros com consistência física por dezenas de passos.
  **Planeje ações** para atingir metas (p.ex., “fazer o bloco vermelho tocar o alvo”) sem ajustar hiperparâmetros por tarefa.

## Objetivos

- Criar um WorldModel que funcional que permita a interpretação de descrições textuais (prompts) e transformá-las em simulações físicas 2D consistentes e realistas para a devida simulação.
- Elaborar um artigo para a consolidação do conhecimento adquirido ao longo do projeto, contribuindo assim para a disseminação acadêmica e análise científica do tema exposto.
- Permitir a compreensão dos membros sobre novas arquiteturas emergentes em inteligência artificial, de tal maneira a ampliar a cpaacidade de inovação do grupo e permitir a criação de oportunidades dos conhecimentos adquiridos nas áreas de educação, pesquisa e desenvolvimento de tecnologias.

## Limites do Projeto

Treinar e avaliar um World Model capaz de compreender e reproduzir a cinemática e dinâmica de objetos em simulações 2D. O projeto não contempla simulações 3D ou aplicações diretas em ambientes físicos, mantendo assim o foco em ambientes digitais controlados.

## Fora do Escopo

- Aplicação em ambiente 3D.

- Aprendizado por reforço end-to-end em hardware físico.

- Contato deformável/fluídos complexos; multiagente competitivo.

## Usuarios/Stakeholders

- Membros da equipe de desenvolvimento e pesquisa diretamente envolvidos no projeto.

- Comunidade acadêmica e técnica interessada em _World Models_, através do artigo de revisão que será publicado.

## Requisitos Funcionais

### Ambiente de simulação deve ser capaz de:

- Gerar vídeos 2D, em formato (.MP4 ou .GIF ou .MKV), representando assim a dinâmica dos objetos a part ir do modelo treinado.

-Receber prompts em **linguagem natural** especificando os elementos, suas propriedades (massa,cor,formato, peso, etc.), condições iniciais (posição, velocidade, energia potencial, etc.) e interações da simulação.

- Permitir que múltiplos cenários com diferentes configurações de objetos sejam executados.

- Registrar automaticamente os experimentos realizados, salvando também os resultados e metadados.

### Modelo treinado deve ser capaz de:

- Ser capaz de inferir a cinemática (posição,velocidade e aceleração) e a dinâmica (forças,colisões, interações) dos objetos simulados.

-Generalizar para diferentes cenários, não apenas aqueles contidos durante o treinamento, e que sigam as mesmas leis físicas.

- Disponibilizar métricas de desempenho (erro médio de previsão de trajetória)

- Ser capaz de interpretar e simular interações entre 2 a N objetos simultâneos, onde N <= 3.

## Métricas de Sucesso

- Métrica de previsão de trajetória (chance de seguir a rota calculada);

## Tópicos que ainda precisam ser abordados na documentação:

- Descrição da arquitetura, técnicas utilizadas, justificativa das escolhas.

- Inserção tutoriais de uso para reprodutibilidade (execução do ambiente, uso de modelo e exemplos de prompts).

## Requisitos não funcionais

Requisitos não funcionais fazem referência aos requisitos que não são intrísecos às funcionalidades do software em si, mas são mais focadas

### Infraestrutura

- O modelo deve ser treinável e executável na infraestrutura computacional que é fornecida no laboratório (CPU/GPU local).

- Deve possuir uma versão reduzida/light que permita a execução em máquinas com recursos mais limitados para testes.

### Qualidade

- Ambiente de simulação deve apresentar consistência visual e física, sem apresentar falhas críticas que impossibilitem a análise.

- Código deve seguir as boas práticas de engenharia de software (modularidade, versionamento, testes unitários básicos).

### Usabilidade

- Interface de usuário deve ser simples e interativa, sem exigir conhecimentos avançados em programação para um usuário comum utilizar dos cenários mais básicos.

- Os prompts devem ser escritos em linguagem natural clara, sem necessidade de sintaxe complexa.

### Reprodutibilidade

- O repositório do gitHub deve conter as instruções completas de instalação, configuração e execução.

- Os experimentos devem poder ser reproduzidos por terceiros com acesso ao dataset e código.

## Entregáveis

- FrontEnd que permita a interação do usuário com o sistema para captar o seu prompt.
- Vídeo que representa o resultado final da simulação praticado pelo usuário.

## Riscos & Suposições

Poder computacional falho, dificuldades de manejar cargas de atividades da equipe.

### English

# Project Scope - World Foundation Model

## Project Overview

Develop, through 2D physical simulations, a World Foundation Model capable of learning the dynamics and kinematics of objects from visual observations (video/images) and predicting/planning the system’s future evolutions.  
The user describes the scene and the task through a prompt; the system instantiates the simulation, executes/predicts rollouts, and returns video outputs, metrics, and planned actions.

## Problem

- Team members currently lack practical experience with **world model architectures**, making this project necessary to enable them to gain expertise and understand both the theoretical foundations and the practical implementation of generative models applied to physical simulation of object behavior in different user-defined scenarios.
- There is still no unified, data-driven model that can:
  - **Generalize** to new combinations of objects/properties (mass, friction, elasticity) in 2D.
  - **Predict** future states with physical consistency across dozens of steps.
  - **Plan actions** to achieve specific goals (e.g., _“make the red block touch the target”_) without requiring task-specific hyperparameter tuning.

## Objectives

- Develop a functional **World Model** capable of interpreting textual descriptions (prompts) and transforming them into consistent and realistic 2D physical simulations.
- Produce an **academic paper** to consolidate the knowledge acquired throughout the project, contributing to scientific dissemination and analysis of the subject.
- Provide team members with a deeper understanding of emerging **artificial intelligence architectures**, expanding the group’s capacity for innovation and enabling the creation of educational, research, and technological development opportunities.

## Project Boundaries

- Train and evaluate a World Model capable of understanding and reproducing the kinematics and dynamics of objects in 2D simulations.
- The project **does not cover** 3D simulations or direct applications in physical environments, maintaining its focus on controlled digital environments.

## Out of Scope

- Application in **3D environments**.
- **End-to-end reinforcement learning** directly in hardware.
- **Deformable contact/complex fluids**; **competitive multi-agent** settings.

## Users / Stakeholders

- Members of the development and research team directly involved in the project.
- The academic and technical community interested in **World Models**, through the review article to be published.

## Functional Requirements

### The simulation environment must be able to:

- Generate 2D videos in formats such as `.MP4`, `.GIF`, or `.MKV`, representing object dynamics based on the trained model.
- Receive **natural language prompts** specifying elements, their properties (mass, color, shape, weight, etc.), initial conditions (position, velocity, potential energy, etc.), and interactions within the simulation.
- Allow multiple scenarios with different object configurations to be executed.
- Automatically log conducted experiments, saving results and metadata.

### The trained model must be able to:

- Infer **kinematics** (position, velocity, acceleration) and **dynamics** (forces, collisions, interactions) of simulated objects.
- **Generalize** to different scenarios, not limited to those present in training, provided they follow the same physical laws.
- Provide **performance metrics**, such as average trajectory prediction error.
- Interpret and simulate interactions between **2 to N simultaneous objects**, where `N <= 3`.

## Success Metrics

- Trajectory prediction metric (probability of following the calculated path).

## Documentation Topics Still to be Addressed

- Description of the architecture, techniques used, and rationale for design choices.
- Tutorials for reproducibility (environment setup, model usage, and example prompts).

## Non-Functional Requirements

Non-functional requirements refer to aspects not directly tied to the software’s functional behavior, but rather to quality attributes and constraints.

### Infrastructure

- The model must be trainable and executable on the computational infrastructure available in the lab (local CPU/GPU).
- A **lightweight version** should be available for execution on resource-constrained machines for testing purposes.

### Quality

- The simulation environment must present visual and physical consistency, without critical failures that would prevent analysis.
- The code must follow **software engineering best practices** (modularity, version control, basic unit testing).

### Usability

- The user interface must be simple and interactive, not requiring advanced programming knowledge for basic scenarios.
- Prompts should be written in clear natural language, without complex syntax requirements.

### Reproducibility

- The GitHub repository must contain complete instructions for installation, configuration, and execution.
- Experiments should be reproducible by third parties with access to the dataset and code.

## Deliverables

- **Frontend** that allows users to interact with the system and provide prompts.
- **Video outputs** representing the final result of the simulation generated from the user’s input.

## Risks & Assumptions

- Limited computational resources.
- Potential difficulties managing team workload.

## Referências | References

[Especificação do projeto](https://docs.google.com/document/d/1GqxDtGbsp0xNqUrYcW_h2VUKyDDuHM6sbOMI6WNnzyQ/edit?tab=t.0)
