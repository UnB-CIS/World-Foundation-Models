---

# Lista de Artigos e Plataformas (Ordem Cronológica)

---

**2018 – World Models [ARTIGO_4.pdf]**

* **Autores:** D. Ha e J. Schmidhuber
* **Técnica:** O trabalho combina três peças principais: um **VAE** (Variational Autoencoder) que transforma imagens do ambiente em uma representação comprimida; um **MDN-RNN** (Mixture Density Network - Recurrent Neural Network) que funciona como memória e faz previsões de estados futuros; e um **controlador simples**, otimizado com CMA-ES, que decide as ações.
* **Contribuição:** Foi uma das primeiras tentativas de mostrar que agentes podem “imaginar” o futuro usando modelos de mundo. A ideia central é dividir a complexidade: o VAE e o RNN aprendem o mundo e o controlador foca só em usar esse conhecimento. O trabalho também prova que é possível treinar o controlador inteiramente dentro de simulações geradas pelo próprio modelo, reduzindo o custo de interagir com o ambiente real.

---

**2019 – Dream to Control: Learning Behaviors by Latent Imagination (Dreamer) [ARTIGO_6.pdf]**

* **Autores:** Danijar Hafner et al.
* **Técnica:** O Dreamer é um algoritmo de RL baseado em modelo que aprende um espaço latente do ambiente e depois “imagina” possíveis sequências de ações dentro desse espaço. Ele traz inovações como normalização de retornos e a loss *symexp twohot*, que melhora a estabilidade ao prever recompensas e valores futuros.
* **Contribuição:** Marca a estreia do **Dreamer V1**, que consegue aprender uma grande variedade de tarefas diferentes sem precisar de ajustes finos em cada caso. Em muitos cenários, supera algoritmos clássicos como PPO e até MuZero, mas com uma eficiência de dados e de computação bem maior. Além disso, mostra que não é necessário treinar agentes com dados de especialistas: a própria imaginação do modelo já é suficiente para guiar boas políticas.

---

**2020 – Planning to Explore via Self-Supervised World Models (Plan2Explore) [ARTIGO_3.pdf]**

* **Autores:** O. Rybkin et al.
* **Técnica:** Esse método usa exploração auto-supervisionada: o modelo aprende sozinho a explorar ambientes sem depender de recompensas externas. A ideia é medir a incerteza do modelo de mundo usando um conjunto de redes (ensemble). Quanto maior a variabilidade entre elas, maior a incerteza, e o agente é incentivado a explorar justamente essas regiões.
* **Contribuição:** O Plan2Explore mostra que um agente pode construir um **modelo geral do mundo** apenas explorando, sem nenhuma tarefa pré-definida. Quando uma nova tarefa aparece, o agente já tem conhecimento suficiente para se adaptar rapidamente (zero-shot ou few-shot). Isso representa um grande avanço em como pensamos sobre exploração, já que o aprendizado não precisa esperar recompensas para ser guiado.

---

**2021 – Mastering Atari with Discrete World Models (DreamerV2) [ARTIGO_5.pdf]**

* **Autores:** D. Hafner et al.
* **Técnica:** O DreamerV2 evolui o Dreamer V1 ao introduzir um **RSSM** (Recurrent State-Space Model) com estados **discretos** em vez de contínuos, o que melhora a representatividade. Além disso, o ator e o crítico são treinados em simulações geradas pelo próprio modelo, reduzindo drasticamente a necessidade de interações reais com o ambiente. O ajuste do **KL balanceado** garante estabilidade no aprendizado.
* **Contribuição:** É um marco: o DreamerV2 foi o primeiro agente a atingir **nível humano no Atari** apenas usando modelos de mundo, algo até então considerado quase impossível. Ele supera agentes model-free altamente otimizados, como Rainbow e IQN, além de se aproximar (ou até superar) o MuZero, mas usando muito menos recursos. Prova que modelos de mundo não são só promissores, mas também práticos.

---

**2022 – Deep Learning, Reinforcement Learning, and World Models [ARTIGO_2.pdf]**

* **Autores:** Yutaka Matsuo, Yann LeCun, Maneesh Sahani, Doina Precup, David Silver, Masashi Sugiyama, Eiji Uchibe, Jun Morimoto et al.
* **Técnica:** É uma revisão ampla que conecta diferentes linhas de pesquisa: aprendizado auto-supervisionado para representações hierárquicas e modelos preditivos, RL offline regulado por comportamento (BREMEN), métodos de atualização de política mais gerais (GPE/GPI) e a ideia de recursos de sucessor.
* **Contribuição:** O artigo resume as discussões do simpósio AIBS2020, trazendo uma visão clara de onde DL, RL e modelos de mundo se encontram. É um ponto de referência para entender as **direções futuras** em busca de uma inteligência mais geral, incluindo inspirações vindas do cérebro humano.

---

**2023 – Genie: Generative Interactive Environments [ARTIGO_7.pdf]**

* **Autores:** Ashley Edwards et al.
* **Técnica:** O Genie é um modelo treinado exclusivamente com vídeos, que consegue gerar mundos interativos quadro a quadro. Ele introduz o **Latent Action Model (LAM)** para aprender ações de forma não supervisionada e o **ST-ViViT**, um tokenizer de vídeo baseado em transformers que transforma os vídeos em tokens discretos, preservando a estrutura temporal.
* **Contribuição:** Foi o primeiro modelo de mundo realmente interativo em tempo real (24 fps). Ele consegue criar cenários jogáveis e responsivos às ações do usuário, mantendo consistência ao longo de longas sequências. Representa uma mudança de paradigma: em vez de apenas prever ou planejar, o modelo **gera mundos nos quais é possível interagir ativamente**.

---

**2025 – Genie 3: A New Frontier for World Models [Google DeepMind]**

* **Autores:** Google DeepMind (colaboradores listados nos agradecimentos)
* **Técnica:** Baseado em geração autorregressiva de vídeo, o modelo produz cada quadro considerando tanto a sequência anterior quanto as ações do usuário. Isso garante continuidade e consistência mesmo em interações longas.
* **Contribuição:** O Genie 3 expande o que o Genie de 2023 iniciou: agora a interatividade em tempo real é mais estável e robusta, com mundos dinâmicos ricos e responsivos. Consolida a ideia de que modelos generativos podem servir como **ambientes abertos e adaptativos**, aproximando IA de experiências interativas complexas, como jogos ou simulações físicas.

---

**2025 – Cosmos World Foundation Model Platform for Physical AI [cosmos.pdf]**

* **Autores:** NVIDIA (lista completa no Apêndice A)
* **Técnica:** Cria uma plataforma de WFMs (World Foundation Models) com dois estágios: pré-treinamento massivo em grandes bases de vídeo e pós-treinamento para tarefas específicas. Utiliza tokenizers de vídeo causais (contínuos e discretos), além de modelos baseados em difusão e transformers autorregressivos. Inclui ainda um sistema de segurança (guardrails).
* **Contribuição:** O Cosmos é uma das primeiras iniciativas a fornecer **modelos de mundo em larga escala e abertos**. A plataforma resolve problemas de falta de dados, gera vídeos 3D fisicamente consistentes em tempo real e é altamente aplicável em robótica, navegação autônoma e ambientes virtuais. O diferencial é ser **open-source e open-weight**, democratizando o acesso a WFMs.

---

**2025 – Understanding World or Predicting Future? A Comprehensive Survey of World Models [ARTIGO_1.pdf]**

* **Autores:** Jingtao Ding et al.
* **Técnica:** É a primeira revisão completa que organiza a literatura de modelos de mundo em duas funções principais: (1) entender como o mundo funciona, construindo representações internas, e (2) prever o futuro para apoiar a tomada de decisão.
* **Contribuição:** O survey sistematiza o campo, mostrando o estado atual, os avanços trazidos por LLMs multimodais e geração de vídeo, e as aplicações práticas em áreas como carros autônomos, robótica e simulações sociais. Também aponta os principais desafios: escalabilidade, realismo físico e segurança. Serve como referência para quem quer entrar na área.

---
