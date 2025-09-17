# Aplicações da plataforma Cosmos

**_Authors / Autores: [@gibi177](http://github.com/gibi177), [@figredos](http://github.com/figredos)_**

## English

### Introduction

[The Cosmos paper](https://arxiv.org/html/2501.03575v1) suggests a series of possible applications of the World Foundation Models platform. Here, some of these possible applications are discussed, along with the different models available for such applications.

### Different Cosmos WFM Models

NVIDIA, through the NVIDIA Developer site [NVIDIA Developer](https://developer.nvidia.com/cosmos?hitsPerPage=6), provides a set of pre-trained models for download. They vary in function for world generation and the acceleration of Physical AI. Below are the different models and their functions.

#### Cosmos Predict-2

Our best world foundation model so far—higher fidelity, flexible frame rates and resolutions, fewer hallucinations, and better control over text, objects, and motion in the video.

Generate previews from text in under 4 seconds and up to 30 seconds of future-world video from a reference image or preview. Below is an example of using the model in `Python`:

```python
import torch
from imaginaire.utils.io import save_image_or_video
from cosmos_predict2.configs.base.config_video2world import PREDICT2_VIDEO2WORLD_PIPELINE_2B
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline

# Create the video generation pipeline.

pipe = Video2WorldPipeline.from_config(
config=PREDICT2_VIDEO2WORLD_PIPELINE_2B,
dit_path="checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-720p-16fps.pt",
text_encoder_path="checkpoints/google-t5/t5-11b",
)

# Specify the input image path and text prompt.

image_path = "assets/video2world/example_input.jpg"
prompt = """
A high-definition video captures the precision of robotic welding in an industrial setting.
The first frame showcases a robotic arm, equipped with a welding torch, positioned over a large metal structure.
The welding process is in full swing, with bright sparks and intense light illuminating the scene, creating a vivid display of blue and white hues.
A significant amount of smoke billows around the welding area, partially obscuring the view but emphasizing the heat and activity.
The background reveals parts of the workshop environment, including a ventilation system and various pieces of machinery,
indicating a busy and functional industrial workspace.
As the video progresses, the robotic arm maintains its steady position, continuing the welding process and moving to its left.
The welding torch consistently emits sparks and light, and the smoke continues to rise, diffusing slightly as it moves upward.
The metal surface beneath the torch shows ongoing signs of heating and melting. The scene retains its industrial ambiance,
with the welding sparks and smoke dominating the visual field, underscoring the ongoing nature of the welding operation.
"""

# Run the video generation pipeline.

video = pipe(input_path=image_path, prompt=prompt)

# Save the resulting output video.

save_image_or_video(video, "output/test.mp4", fps=16)
```

For more information on using the model, see the [Cosmos Predict GitHub](https://github.com/nvidia-cosmos/cosmos-predict2?tab=readme-ov-file).

This [article](https://developer.nvidia.com/blog/develop-custom-physical-ai-foundation-models-with-nvidia-cosmos-predict-2/) explains a possible usage pipeline for the model.

#### Cosmos Transfer

A family of highly performant, pre-trained world foundation models designed to generate videos aligned with input control conditions.

The Cosmos Transfer1 models are a collection of diffusion-based world foundation models capable of generating dynamic, high-quality videos from text and control video inputs. They can serve as a foundation for various applications or research related to world generation. The models are ready for commercial use.

For more information on using the model, see the [Cosmos Transfer1 GitHub](https://github.com/nvidia-cosmos/cosmos-transfer1).

#### Cosmos Reason

Physical AI models understand physical common sense and generate appropriate embodied decisions in natural language through long chain-of-thought reasoning processes.

The Cosmos-Reason1 models are tuned with physical common sense and embodied reasoning data, using supervised fine-tuning and reinforcement learning. These are Physical AI models capable of understanding space, time, and fundamental principles of physics, and can serve as planning models to reason about an embodied agent’s next steps.

For more information on using the model, see the [Cosmos Reason1 GitHub](https://github.com/nvidia-cosmos/cosmos-reason1).

#### Cosmos Tokenizers

Cosmos Tokenizer is a set of visual tokenizers for images and videos that offers different compression rates while maintaining high reconstruction quality. It serves as an efficient building block for image and video generation models based on diffusion and autoregressive approaches.

There are two types of tokenizers:

- **Continuous (C)**: Encodes visual data into continuous latent embeddings, as in latent diffusion models (e.g., Stable Diffusion). Ideal for models that generate data by sampling from continuous distributions.
- **Discrete (D)**: Encodes visual data into discrete latent codes, mapping to quantized indices, as in autoregressive transformers (e.g., VideoPoet). Essential for models that optimize cross-entropy loss, such as GPT-like models.

Each type has a variant for images (I) and videos (V):

- **Cosmos-Tokenizer-CI**: Continuous for images
- **Cosmos-Tokenizer-DI**: Discrete for images
- **Cosmos-Tokenizer-CV**: Continuous for videos
- **Cosmos-Tokenizer-DV**: Discrete for videos

Given an image or video, Cosmos Tokenizer produces continuous latent or discrete tokens. It achieves spatial compression rates of 8x8 or 16x16 and temporal factors of 4x or 8x, totalling up to a 2048x compression factor (8x16x16). This is 8x more compression than state-of-the-art methods, while maintaining superior image quality and up to 12x speed over the best tokenizers currently available.

In short, Cosmos Tokenizer combines efficiency, high compression, and quality, making it an advanced solution for generative AI applications involving images and videos.

For more information on using the model, see the [Cosmos Tokenizer GitHub](https://github.com/NVIDIA/Cosmos-Tokenizer).

#### Cosmos WFM Post-Training Samples

Cosmos Sample Models for Autonomous Driving are a family of high-performance Cosmos foundation models, post-trained specifically for autonomous driving scenarios.

These models are fine-tuned versions of the Cosmos World foundation models, capable of generating high-quality, multi-view consistent driving videos from text, image, or video inputs. They serve as versatile building blocks for various applications and research related to autonomous driving. Ready for commercial use, the models are available under the NVIDIA Open Model License Agreement.

For more information on using the model, see the [Cosmos Predict GitHub](https://github.com/nvidia-cosmos/cosmos-predict2?tab=readme-ov-file).

#### Cosmos Guardrails

A family of highly performant, pre-trained world foundation models designed to generate videos and world states with physical awareness for the development of Physical AI. Cosmos Guardrail is a content safety model composed of three components that ensure content safety:

1. Blocklist: An expert-curated keyword list used to filter edge cases and sensitive terms.

2. Video Content Safety Filter: A multi-class classifier trained to distinguish between safe and unsafe frames in generated videos, using SigLIP embeddings for high-accuracy detection of inappropriate content.

3. Face Blur Filter: A pixelation filter based on RetinaFace that identifies facial regions with high confidence and applies pixelation to any detections larger than 20x20 pixels, promoting anonymization and privacy in generated scenes.

These components work together to ensure that both text prompts and generated video content meet the content safety standards required for commercial Physical AI applications.

#### Cosmos Upsampler

Cosmos-1.0-Prompt-Upsampler-Text2World is a large language model (LLM) designed to transform original prompts into more detailed and enriched versions. It enhances prompts by adding information and maintaining a consistent descriptive structure before they are used in a text-to-world model, which typically results in higher-quality outputs. This model is ready for commercial use.

### Uses of Cosmos WFM

Below are some of the different applications of the platform.

#### Training autonomous cars

A number of companies in the transportation sector have adopted the Cosmos WFM platform for **_Autonomous Vehicles (AV)_** solutions.

- **_Waabi_**, a pioneer in generative AI for the physical world, starting with autonomous vehicles, is evaluating Cosmos in the context of data curation for AV software development and simulation.

- **_Wayve_**, a company developing AI foundation models for autonomous driving, is evaluating Cosmos as a tool to research corner-case driving scenarios used for safety and validation.

- **_Uber_**, the global ride-sharing giant, is partnering with NVIDIA to accelerate autonomous mobility. Uber’s joint driving data assets, combined with capabilities from the Cosmos platform and NVIDIA DGX Cloud, can help AV partners build stronger AI models even more efficiently.

This [article](https://developer.nvidia.com/blog/simplify-end-to-end-autonomous-vehicle-development-with-new-nvidia-cosmos-world-foundation-models/) shows a way to simplify end-to-end autonomous vehicle development with the _Cosmos WFM_ platform.

#### Synthetic dataset generation

In recent years, with advances in computer vision and deep learning models, there has been a strong demand for large volumes of data to train these networks. In this context, one of the most promising applications of Cosmos is precisely the generation of synthetic data, especially where collecting real data is costly or unfeasible. Thus, Cosmos acts as an artificial extender of existing datasets, meeting the demand for trainable data mentioned above.

![Pedestrian Cross-walk](images/pedestrian_crosswalk.webp)

_As an example, based on a short video or image showing a pedestrian crossing at a crosswalk, it is possible to simulate different weather conditions, lighting, times of day, different angles, etc. This diversity is very useful for training autonomous vehicles, for instance, as it greatly reduces the size of the real video training dataset._

In this sense, **Cosmos Predict** proves to be the most suitable model, as its purpose is data generation itself.

Additionally, an important discussion to be had is the **feasibility** of the application. For video generation, one of the simplest models is _Cosmos-Predict2-2B-Video2World_, with 2 billion parameters. Although this model is the simplest released by NVIDIA, it does not run locally on notebooks. Perhaps with a more powerful GPU it is possible to run it with limitations, but in general the computational cost is higher than a typical notebook can support. Thus, the alternative of using cloud computing arises, which also comes with associated financial costs.

| Input image                              | Output video                       |
| ---------------------------------------- | ---------------------------------- |
| ![Input Image](images/example_input.jpg) | ![Output video](images/output.mp4) |

_Example of how Cosmos-Predict2-2B-Video2World can be used. With an image and text as input, a short video is generated. This strategy can be replicated for different situations and applications, such as the crosswalk example mentioned earlier._

Another relevant point is the quality of **synthetic data**. Although visually realistic, this data may still contain biases or inconsistencies that affect model training. Therefore, it is necessary to use practices for validating generated outputs and comparing them with real reference data. This verification can be done with **Cosmos-Reason**, highlighted earlier for its capability to interpret videos, or even manually, depending on the size of the synthetic dataset.

#### Other Applications

Using the same approach as the previous application—where _Cosmos-Predict2-2B-Video2World_ is used to generate videos from input images or videos combined with a text prompt—opens the door to several other similar applications. Among them, the following stand out:

- **Scene pre-visualization in films and games**: In the creative industry, the production of films, animations, and digital games goes through various stages of visual prototyping. Traditionally, this requires using 3D modeling tools, physical simulation, and rendering—processes that can be expensive and time-consuming. With Cosmos, however, it is possible to create previsualizations of entire scenes using only simple sketches and textual descriptions. For example, an artist can submit a photo of a landscape and add something like “strong wind swaying the trees at dusk.” The model then generates a short clip that shows this scene with movement, lighting, and weather, without needing to go through the modeling stage.

- **Reconstruction of historical scenes for museums**: Cultural centers can also benefit greatly from using Cosmos. With it, it becomes possible to reconstruct scenes from the past from paintings or old photographs, allowing visitors greater engagement with the content on display and expanding their understanding of the historical context.

## Portuguese

### Introdução

O artigo da [Cosmos](https://arxiv.org/html/2501.03575v1), sugere uma série de possíveis aplicações da plataforma de World Foundation Models, aqui são discutidas algumas dessas possíveis aplicações, além dos diferentes modelos disponibilizados para tais aplicações.

### Diferentes Modelos do Cosmos WFM

A NVIDIA, através do site [nvidia Developer](https://developer.nvidia.com/cosmos?hitsPerPage=6), disponibiliza uma série de modelos pré-treinados para download. Eles variam em função, para geração de mundo, e aceleração de IA física. Abaixo estão listados os diferentes modelos e suas funções.

#### Cosmos Predict-2

Nosso melhor modelo fundamental de mundo até agora—maior fidelidade, taxas de quadros e resoluções flexíveis, menos alucinações e melhor controle de texto, objetos e movimento no vídeo.

Gere prévias a partir de texto em menos de 4 segundos e até 30 segundos de vídeo do mundo futuro a partir de uma imagem de referência ou prévia. Abaixo é mostrada a utilização do modelo em `python`:

```python
import torch
from imaginaire.utils.io import save_image_or_video
from cosmos_predict2.configs.base.config_video2world import PREDICT2_VIDEO2WORLD_PIPELINE_2B
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline

# Create the video generation pipeline.

pipe = Video2WorldPipeline.from_config(
config=PREDICT2_VIDEO2WORLD_PIPELINE_2B,
dit_path="checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-720p-16fps.pt",
text_encoder_path="checkpoints/google-t5/t5-11b",
)

# Specify the input image path and text prompt.

image_path = "assets/video2world/example_input.jpg"
prompt = """
A high-definition video captures the precision of robotic welding in an industrial setting.
The first frame showcases a robotic arm, equipped with a welding torch, positioned over a large metal structure.
The welding process is in full swing, with bright sparks and intense light illuminating the scene, creating a vivid display of blue and white hues.
A significant amount of smoke billows around the welding area, partially obscuring the view but emphasizing the heat and activity.
The background reveals parts of the workshop environment, including a ventilation system and various pieces of machinery,
indicating a busy and functional industrial workspace.
As the video progresses, the robotic arm maintains its steady position, continuing the welding process and moving to its left.
The welding torch consistently emits sparks and light, and the smoke continues to rise, diffusing slightly as it moves upward.
The metal surface beneath the torch shows ongoing signs of heating and melting. The scene retains its industrial ambiance,
with the welding sparks and smoke dominating the visual field, underscoring the ongoing nature of the welding operation.
"""

# Run the video generation pipeline.

video = pipe(input_path=image_path, prompt=prompt)

# Save the resulting output video.

save_image_or_video(video, "output/test.mp4", fps=16)
```

Para mais informações sobre a utilização do modelo, utilizar [github do cosmos predict](https://github.com/nvidia-cosmos/cosmos-predict2?tab=readme-ov-file).

Esse [artigo](https://developer.nvidia.com/blog/develop-custom-physical-ai-foundation-models-with-nvidia-cosmos-predict-2/) explica uma possível pipeline de utilização do modelo.

#### Cosmos Transfer

Uma família de modelos fundamentais de mundo pré-treinados altamente performáticos, projetados para gerar vídeos alinhados com as condições de controle de entrada.

Os modelos Cosmos Transfer1 são uma coleção de modelos fundamentais de mundo baseados em difusão, capazes de gerar vídeos dinâmicos e de alta qualidade a partir de texto e entradas de vídeo de controle. Eles podem servir como base para diversas aplicações ou pesquisas relacionadas à geração de mundos. Os modelos estão prontos para uso comercial.

Para mais informações sobre a utilização do modelo, utilizar [github do cosmos tranfer1](https://github.com/nvidia-cosmos/cosmos-transfer1).

#### Cosmos Reason

Modelos de IA Física compreendem o senso comum físico e geram decisões corporificadas apropriadas em linguagem natural por meio de longos processos de raciocínio em cadeia.

Os modelos **_Cosmos-Reason1_** são ajustados com dados de senso comum físico e raciocínio corporificado, utilizando afinação supervisionada e aprendizado por reforço. Estes são modelos de IA Física capazes de entender espaço, tempo e princípios fundamentais da física, podendo servir como modelos de planejamento para raciocinar sobre os próximos passos de um agente corporificado.

Para mais informações sobre a utilização do modelo, utilizar [github do cosmos reason1](https://github.com/nvidia-cosmos/cosmos-reason1).

#### Cosmos Tokenizers

O Cosmos Tokenizer é um conjunto de tokenizadores visuais para imagens e vídeos que oferece diferentes taxas de compressão, mantendo alta qualidade de reconstrução. Ele serve como um bloco eficiente para modelos de geração de imagens e vídeos baseados em difusão e em abordagens autoregressivas.

Existem dois tipos de tokenizadores:

- **Contínuo (C)**: Codifica os dados visuais em embeddings latentes contínuos, como em modelos de difusão latente (exemplo: Stable Diffusion). Ideal para modelos que geram dados amostrando de distribuições contínuas.
- **Discreto (D)**: Codifica os dados visuais em códigos latentes discretos, mapeando para índices quantizados, como em transformadores autoregressivos (exemplo: VideoPoet). Essencial para modelos que otimizam a perda de entropia cruzada, como os modelos GPT.

Cada tipo possui variação para imagens (I) e vídeos (V):

- **Cosmos-Tokenizer-CI**: Contínuo para imagens
- **Cosmos-Tokenizer-DI**: Discreto para imagens
- **Cosmos-Tokenizer-CV**: Contínuo para vídeos
- **Cosmos-Tokenizer-DV**: Discreto para vídeos

Dado uma imagem ou vídeo, o Cosmos Tokenizer gera latentes contínuos ou tokens discretos. Ele atinge taxas espaciais de compressão de 8x8 ou 16x16 e fatores temporais de 4x ou 8x, somando até um fator total de compressão de 2048x (8x16x16). Isso é 8x mais compressão que métodos de ponta, mantendo qualidade superior de imagem e velocidade até 12x maior que os melhores tokenizadores disponíveis atualmente.

Em resumo, Cosmos Tokenizer combina eficiência, alta compressão e qualidade, sendo uma solução avançada para aplicações em inteligência artificial generativa envolvendo imagens e vídeos.

Para mais informações sobre a utilização do modelo, utilizar [github do cosmos tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer).

#### Cosmos WFM Post-Training Samples

Modelos Cosmos Sample para Condução Autônoma são uma família de modelos fundamentais Cosmos de alto desempenho, pós-treinados especialmente para cenários de condução autônoma.

Esses modelos são versões ajustadas dos modelos fundamentais Cosmos World, capazes de gerar vídeos de condução de alta qualidade e consistentes em múltiplas vistas a partir de entradas de texto, imagem ou vídeo. Servem como blocos versáteis para diversas aplicações e pesquisas relacionadas à condução autônoma. Prontos para uso comercial, os modelos estão disponíveis sob o Acordo de Licença de Modelos Abertos da NVIDIA.

Para mais informações sobre a utilização do modelo, utilizar [github do cosmos predict](https://github.com/nvidia-cosmos/cosmos-predict2?tab=readme-ov-file).

#### Cosmos Guardrails

Uma família de modelos fundamentais de mundo pré-treinados e altamente performáticos, projetados para gerar vídeos e estados de mundo com consciência física para o desenvolvimento de IA física.
O Cosmos Guardrail é um modelo de segurança de conteúdo composto por três componentes que garantem a segurança do conteúdo:

1. Blocklist: Uma lista de palavras-chave selecionadas por especialistas utilizada para filtrar casos extremos e termos sensíveis.

2. Video Content Safety Filter: Um classificador de múltiplas classes treinado para distinguir entre quadros seguros e inseguros em vídeos gerados, utilizando embeddings SigLIP para alta precisão na detecção de conteúdo inadequado.

3. Face Blur Filter: Um filtro de pixelização baseado no RetinaFace, que identifica com alta confiança regiões faciais e aplica pixelização em quaisquer detecções maiores que 20x20 pixels, promovendo anonimização e privacidade nas cenas geradas.

Esses componentes atuam de forma integrada para garantir que tanto os prompts de texto quanto o conteúdo de vídeo gerado atendam aos padrões de segurança de conteúdo necessários para aplicações comerciais em IA física

#### Cosmos Upsampler

O Cosmos-1.0-Prompt-Upsampler-Text2World é um modelo de linguagem de grande porte (LLM) projetado para transformar prompts originais em versões mais detalhadas e enriquecidas. Ele aprimora os prompts adicionando informações e mantendo uma estrutura descritiva consistente antes que sejam utilizados em um modelo text-to-world, o que normalmente resulta em saídas de maior qualidade. Este modelo está pronto para uso comercial.

### Usos do Cosmos WFM

Abaixo são citadas algumas das diferentes aplicações da plataforma.

#### Treinamento de Carros autônomos

Uma série de empresas do setor de transportes adotaram a plataforma _Cosmos WFM_ para soluções de **_AV (Autonomous Vehicles)_**

- **_Waabi_**, empresa pioneira em IA generativa para o mundo físico, começando com veículos autônomos, está avaliando o Cosmos no contexto da curadoria de dados para desenvolvimento e simulação de software de _AV_.

- **_Wayve_**, empresa que está desenvolvendo modelos de base de IA para direção autônoma, está avaliando o Cosmos como uma ferramenta para pesquisar cenários de direção em curvas e esquinas usados para segurança e validação.

- **_Uber_**, a gigante global de compartilhamento de viagens está fazendo uma parceria com a NVIDIA para acelerar a mobilidade autônoma. Os riscos conjuntos de dados de direção da Uber, combinados com recursos da plataforma Cosmos e do NVIDIA DGX Cloud, podem ajudar os parceiros de _AV_ a criarem modelos de IA mais fortes de forma ainda mais eficiente.

Esse [artigo](https://developer.nvidia.com/blog/simplify-end-to-end-autonomous-vehicle-development-with-new-nvidia-cosmos-world-foundation-models/) mostra uma forma de simplificar desenvolvimento end-to-end de veículos autônomos com a plataforma _Cosmos WFM_.

#### Geração de datasets sintéticos

Nos últimos anos, com o avanço dos modelos de visão computacional e deep learning, surgiu uma forte demanda por grandes volumes de dados para o treinamento dessas redes. Nesse cenário, uma das aplicações mais promissoras do Cosmos é justamente a geração de dados sintéticos, especialmente em contextos onde a coleta de dados reais é custosa, ou inviável. Assim, o Cosmos entra como um extensor artificial de datasets já existentes, suprindo a demanda por dados treináveis mencionada.

![Faixa de pedestres](images/pedestrian_crosswalk.webp)

> Como exemplo, com base em um vídeo curto ou imagem mostrando um pedestre atravessando na faixa, é possível simular diferentes condições climáticas, de iluminação, de horários, com ângulos diferentes, etc. Essa diversidade vem a ser muito útil para o treinamento de veículos autônomos, por exemplo, já que reduz muito o tamanho do dataset de vídeos reais de treinamento.

Nesse sentido, o **Cosmos Predict** se mostra como o modelo mais adequado, já que seu intuito é a própria geração de dados.

Ademais, uma discussão importante a ser feita é sobre a **viabilidade** da aplicação. Para a geração de vídeos, um dos modelos mais simples é o _Cosmos-Predict2-2B-Video2World_, com 2 Bilhões de parâmetros. Esse modelo, embora seja o mais simples disponibilizado pela Nvidia, não roda localmente em notebooks. Talvez com uma GPU mais potente seja possível rodar com limitações, mas de forma geral o custo computacional é mais alto do que um notebook normal suporta. Assim, abre-se a alternativa de usar cloud computing, que também vem com custos financeiros associados.

| Input image                              | Output video                       |
| ---------------------------------------- | ---------------------------------- |
| ![Input Image](images/example_input.jpg) | ![Output video](images/output.mp4) |

> Exemplo de como o _Cosmos-Predict2-2B-Video2World_ pode ser usado. Com uma imagem e texto de input, gera-se um vídeo curto. Essa estratégia pode ser simulada para diferentes situações e aplicações, como por exemplo a da faixa de pedestres mencionada anteriormente.

Outro ponto relevante é a **qualidade dos dados sintéticos**. Embora visualmente realistas, esses dados ainda podem conter vieses ou inconsistências que afetam o treinamento dos modelos. Dessa forma, faz-se necessário o uso de práticas de validação dos outputs gerados e comparação com dados reais de referência. Essa verificação pode ser feita com o **Cosmos-Reason**, destacado anteriormente pela sua capacidade interpretativa de vídeos, ou mesmo manualmente, a depender do tamanho do dataset sintético.

#### Outras Aplicações

Usar a mesma abordagem da aplicação anterior, em que usamos o _Cosmos-Predict2-2B-Video2World_ para gerar vídeos a partir de imagens ou vídeos de entrada somados a um input de texto, abre portas para várias outras aplicações de cunho semelhante. Dentre elas, podemos destacar:

- **Pré-visualização de cenas em filmes e jogos:** Na indústria criativa, a produção de filmes, animações e jogos digitais passa por várias etapas de prototipação visual. Tradicionalmente, isso exige o uso de ferramentas de modelagem 3D, simulação física e renderização, processos que podem ser caros e demorados. Com o Cosmos, porém, é possível criar pré-visualizações de cenas inteiras, usando apenas esboços e descrições textuais simples. Por exemplo, um artista pode enviar uma foto de uma paisagem e adicionar algo como "vento forte balançando as árvores ao entardecer". O modelo então gera um clipe curto que mostra essa cena com movimento, iluminação e clima, sem precisar passar pel etapa de modelagem.

- **Reconstrução de cenas históricas para museus:** Centros culturais também podem se beneficiar muito do uso do Cosmos. Com ele, torna-se possível reconstruir cenas do passado a partir de pinturas ou fotografias antigas, o que permite ao visitante um engajamento maior com o conteúdo exposto e amplia a sua compreensão do contexto histórico.

### Referências

- [NVIDIA Cosmos for Developers](https://developer.nvidia.com/cosmos?hitsPerPage=6)
- [World Foundation Models: 10 Use Cases & Examples [2025]](https://research.aimultiple.com/world-foundation-model/)
