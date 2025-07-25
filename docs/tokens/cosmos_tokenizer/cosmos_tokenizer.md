# Cosmos Tokenizer

## Português

### Visão Geral

Tokenizers são blocos fundamentais na construção de modelos modernos em larga escala. Eles transformam dados brutos em representações mais eficientes ao aprender um espaços latentes "bottle-necked" descobertos de maneira não supervisionada. Especificamente, tokenizers visuais mapeiam dados visuais brutos e redundantes em tokens semânticos compactos, o que os torna cruciais para lidar com dados visuais de alta dimensionalidade.

> Mapear os dados brutos (no formato de valores de pixels) para um "espaços latentes bottle-necked" significa que a imagem original — que é muito alta em dimensionalidade (para uma pequena imagem RGB com dimensões $224\times 224 \times 3$ você tem um total de $196.608$ características por imagem) — será comprimida em uma forma menor e mais útil ao aprender uma representação interna comprimida (espaço latente).

> Tudo isso para dizer que as imagens serão comprimidas para uma forma de menor dimensionalidade (tokens) ao passar por um _tokenizer_ que é treinado de forma não supervisionada.

A imagem abaixo ilustra o pipeline de treinamento de tokenização, onde o objetivo é treinar o codificador (encoder) e o decodificador (decoder), de forma que a representação por tokens no gargalo preserve ao máximo a informação visual do input.

![Tokenization Pipeline](../images/cosmos_tokenizer/tokenization_pipeline.png)

Na pipeline, um vídeo de entrada é codificado em tokens, que geralmente são muito mais compactos do que o vídeo de entrada. O decodificador então reconstrói o vídeo original a partir desses tokens. _O treinamento do tokenizer consiste em aprender o codificador e decodificador de modo a preservar ao máximo a informação visual nos tokens_.

Tokenizers existem em dois tipos: contínuos e discretos. Tokenizers contínuos codificam dados visuais em embeddings contínuos latentes, como nos modelos de difusão latente (latent diffusion models) tal como o _Stable Diffusion_ ou o _VideoLDM_. Esses embeddings são adequados para modelos que geram dados ao amostrar de distribuições contínuas. Tokenizers discretos codificam dados visuais em códigos latentes discretos, mapeando-os para índices quantizados, como visto em transformers autorregressivos como o VideoPoet. Essa representação discreta é necessária para modelos como o GPT, que são treinados com _cross-entropy loss_.

> _**Tokenizers contínuos**_: codificam os dados em um espaço vetorial contínuo e de alta dimensionalidade. Eles são usados em modelos de difusão, pois esses modelos geram dados por meio de amostragem em distribuições contínuas. Esses embeddings permitem ao modelo interpolar e reamostrar variações nos dados de base.

> _**Tokenizers discretos**_: codificam dados em códigos latentes discretos, que são quantizados ou mapeados para um conjunto de índices finitos distintos. Esses tokenizers são usados com _modelos autorregressivos_ que geram sequências um token por vez. O artigo cita os _modelos GPT_ e como eles são treinados com cross-entropy loss; essa abordagem requer tokens discretos, pois trata o processo de geração como uma predição sobre um vocabulário fixo, sendo essa função de perda voltada a medir a diferença entre distribuições categóricas previstas e reais.

> A principal diferença entre os dois tipos de tokenizers está na forma como os tokenizers discretos mapeiam os valores de imagem para valores discretos ($\mathbb{N}$), enquanto tokenizers contínuos mapeiam para valores reais ($\mathbb{R}$), permitindo uma maior quantidade de valores no espaço latente (por exemplo, _"[...]espaço vetorial de alta dimensionalidade[...]”_).

> Modelos de difusão aprendem ao reverter um processo gradual de "adicionar ruído" a dados reais. Esse processo gradual é o motivo pelo qual tais modelos dependem de tokens com valores reais ($\mathbb{R}$).

O sucesso dos tokenizers depende, em grande parte, da sua habilidade de fornecer altas taxas de compressão sem comprometer a qualidade da reconstrução visual posterior. Por um lado, uma alta compressão reduz os requisitos de armazenamento e computação. Por outro, uma compressão excessiva pode levar à perda de detalhes visuais essenciais. Esse equilíbrio representa um desafio importante no projeto de tokenizers.

A imagem a seguir ilustra os dois tipos de tokens:

![Visualization of continuous and discrete tokenizers](../images/cosmos_tokenizer/token_types.png)

Tokens ao longo das dimensões espaciais ($\frac{H}{S_{HW}} \times \frac{W}{S_{HW}}$) e temporais ($1 + \frac{T}{S_T}$), com um fator de compressão espacial $S_{HW}$ e um fator de compressão temporal $S_T$. O primeiro token temporal representa o primeiro quadro da entrada, possibilitando a tokenização conjunta de imagens ($T=0$) e vídeos ($T>0$) em um espaço latente compartilhado.

> $S_{HW}$ é o _fator de compressão espacial_ usado para comprimir as dimensões espaciais de uma imagem. Essa é uma etapa chave no processo de tokenização espacial, no qual o quadro de entrada é dividido em pequenos blocos ou regiões, cada um dos quais é representado por um ou mais tokens.

> Se a dimensão da imagem original for $224\times 224 \times 3$, e o _fator de compressão espacial_ for 16, a grade de tokens será de $14\times 14$, e cada token conterá informações sobre uma região de $16\times 16$ pixels.

> $S_T$ é o _fator de compressão temporal_, e é usado para reduzir o número de tokens que representam o eixo temporal agrupando quadros. Isso é aplicado no processo de _Tokenização Temporal_, para representar o número de quadros. A adição de $1$ permite ao modelo tratar o quadro inicial como um token especial, dando suporte à tokenização conjunta de imagens e vídeos. Se o processo for aplicado a uma imagem, $T=0$ e a dimensão temporal reduz-se a $1$.

> Se a imagem mencionada acima fizer parte de um vídeo com $32$ quadros, e o _fator de compressão temporal_ tiver valor $4$, o processo de tokenização produzirá 9 tokens temporais: $1$ para o primeiro quadro (adaptabilidade para lidar com imagens), e outros 8, cada um agrupando 4 quadros.

A tabela a seguir ilustra diferentes tokenizers visuais e suas capacidades:

![Different tokenizers and capabilities](../images/cosmos_tokenizer/tokenizers_table.png)

O _Tokenizador Cosmos_ utiliza uma arquitetura leve e computacionalmente eficiente com um mecanismo temporal causal. Especificamente, ele emprega camadas de convolução temporal causal e camadas de atenção temporal causal para preservar a ordem temporal natural dos quadros de vídeo.

> O termo "causal" implica que qualquer predição sobre um determinado quadro ou ponto no tempo é baseada somente nesse quadro e em todos os quadros anteriores, nunca nos futuros. Portanto, "_Convolução Temporal Causal_" significa que a geração de características para um dado quadro utiliza apenas dados do quadro $t$ para trás.

> A mesma ideia aplica-se à "_Atenção Temporal Causal_", onde o tokenizer pondera dinamicamente em quais quadros focar ao tomar decisões sobre o quadro atual.

Os tokenizers são treinados diretamente em imagens de alta resolução e vídeos de longa duração, sem limitar as categorias ou proporções de aspecto. O _Cosmos Tokenizer_ opera em diferentes proporções de aspecto. Ele é agnóstico quanto à duração temporal durante a inferência, sendo capaz de tokenizar além da duração temporal usada durante o treinamento.

Os gráficos abaixo mostram a comparação de desempenho entre o _Cosmos Tokenizer_ e outros tokenizers, evidenciando a sua qualidade superior mesmo em taxas de compressão mais altas:

![Tokenizer comparisons](../images/cosmos_tokenizer/tokenizer_comparison.png)

Claro! Segue abaixo o texto traduzido para o português, com a estrutura Markdown totalmente preservada, e mantendo em inglês apenas os termos técnicos que fazem mais sentido no original:

### Arquitetura

O Cosmos Tokenizer é projetado com uma arquitetura encoder-decoder. Dado um vídeo de entrada $x_{0:T} \in \mathbb{R^{(1 + T) \times H \times W \times 3}}$, com $H,\ W,\ T$ representando a altura, largura e número de quadros, o encoder ($\varepsilon$) tokeniza as entradas em um vídeo de tokens $z_{0:T'} \in \mathbb{R^{(1 + T) \times H \times W \times 3}}$, com um fator de compressão espacial de $s_{H W} = \frac{H}{H'}=\frac{W}{W'}$ e um fator de compressão temporal de $S_T = \frac{T}{T'}$. O decoder ($\mathcal{D}$) então reconstrói o vídeo de entrada a partir desses tokens, resultando no vídeo reconstruído $\hat{x}_{0:T} \in \mathbb{R^{(1 + T) \times H \times W \times 3}}$

$$\hat{x}_{0:T} = \mathcal{D}(\varepsilon(x_{0:T}))$$

> Esta é uma visão geral da arquitetura, onde o encoder codifica uma entrada $x_{0:T}$ em tokens $z_{0:T'}$, e o decoder decodifica esses tokens e produz $\hat{x}_{0:T}$.

Nossa arquitetura emprega um design temporalmente causal, garantindo que cada estágio processe apenas quadros atuais e passados. _Nosso tokenizer opera no wavelet space, onde as entradas são primeiro processadas por uma wavelet transform de 2 níveis_. A wavelet transform mapeia o vídeo de entrada $x_{0:T}$ de forma agrupada para realizar um downsample das entradas por um fator de quatro ao longo das direções $x, y$ e $t$. Os grupos são formados como: $\lbrace x_0, x_{1:4}, x_{5:8}, ..., x_{(T-3):T}\rbrace \rightarrow \lbrace g_0, g_1, g_2, ..., g_{T/4}\rbrace$. Estágios subsequentes do encoder processam esses quadros de forma temporalmente causal como $\lbrace g_0, g_{0:1}. g_{0:2}, ...\rbrace \rightarrow \lbrace \xi_0, \xi_1, \xi_2,...  \rbrace$. Estágios posteriores seguem um esquema similar, produzindo finalmente os tokens $z_{0:T'}$.

> A _**wavelet transform**_ é uma técnica para processamento de sinais em múltiplas escalas e resoluções. Ela difere de transformadas mais tradicionais, como a _Fourier Transform_, que representam dados em termos de ondas senoidais e cossenoidais de frequência fixa, utilizando no lugar oscilações curtas e semelhantes a ondas, que podem ser escaladas e deslocadas. Esta transformada decompõe tanto variações espaciais quanto temporais (ao longo dos quadros), comprimindo e isolando mudanças bruscas em regiões suaves.

> O **_wavelet space_** é a representação de um sinal após passar por uma wavelet transform. Assim, a sentença "_[...]nosso tokenizer opera no wavelet space, onde as entradas são primeiro processadas por uma wavelet transform de 2 níveis[...]_" significa que cada dimensão espacial e temporal é decomposta, extraindo informações tanto de baixa frequência (globais) quanto de alta frequência.

> Durante o processo de aplicação da wavelet transform de 2 níveis, os dados são redimensionados em cada dimensão ($x, y, t$) por um fator de $4$. Assim, cada grupo de $4$ pixels ($x_{t:(t+3)}$) nas 3 dimensões da imagem é representado por um grupo comprimido ($g_i$).

> O tokenizer final utiliza a Haar Wavelet, que é uma das funções wavelet mais simples.

> ![haar_wavelet](../images/cosmos_tokenizer/haar_wavelet.png)
>
> Wavelet Transforms comprimem imagens por meio de decomposição, primeiro em uma aproximação de baixa resolução da imagem original, em seguida com detalhes verticais, horizontais, e diagonais, semelhante ao demonstrado abaixo:

> ![Wavelet decomposition](../images/cosmos_tokenizer/wavelet_decomposition.png)

> Resultando em imagens comprimidas como esta:

> ![Wavelet compressed image](../images/cosmos_tokenizer/wavelet_compressed_image.png)

E uma transformada wavelet de 2 níveis teria uma aparência semelhante à seguinte:

> ![2-level wavelet](../images/cosmos_tokenizer/2-level_wavelet.png)

O design causal ajuda a adaptar modelos baseados no tokenizer para aplicações de Physical AI que frequentemente operam em cenários temporalmente causais. A wavelet transform nos permite operar sobre uma representação de vídeo mais compacta que elimina redundâncias na informação de pixel, permitindo que as camadas restantes foquem em compressão mais semântica.

Nossos estágios de encoder são implementados utilizando uma série de residual blocks intercalados com downsampling blocks. Em cada bloco, utilizamos uma convolução 3D fatorada espaço-temporalmente, onde aplicamos primeiro uma convolução 2D com kernel de tamanho $1\times k\times k$ para capturar informações espaciais, seguida por uma convolução temporal com kernel de tamanho $k\times 1\times 1$ para capturar dinâmicas temporais. Utilizamos padding à esquerda de k-1 para garantir causalidade.

> Emprega convolução (2 + 1)D.

Para capturar dependências de longo alcance, utilizamos uma self-attention causal fatorada espaço-temporal com uma global support region. Usamos a função de ativação Swish para não-linearidade. Utilizamos Layer Normalization (LayerNorm) em vez de Group Normalization (GroupNorm), o que evita o aparecimento de grandes magnitudes em regiões específicas do espaço latente ou das saídas reconstruídas. O decoder espelha o encoder, substituindo os downsampling blocks por um upsampling block. A imagem abaixo mostra uma visão geral da arquitetura do Cosmos Tokenizer.

> Global support region para não-linearidade significa que os tokens interagem com todos os outros tokens disponíveis no momento (devido às restrições da arquitetura causal).

> A **_Swish activation function_**, definida por $\operatorname{Swish}^{\beta}(x) = x \cdot sigmoid(\beta x) = \frac{x}{1+e^{-\beta x}}$

> ![Swish activation function](../images/cosmos_tokenizer/swish_activation_function.png)

![Tokenizer architecture](../images/cosmos_tokenizer/tokenizer_architecture.png)

A imagem mostra a **arquitetura geral do Cosmos Tokenizer, ilustrando a integração da causalidade temporal com a estrutura encoder-decoder.** A causalidade temporal (à esquerda) processa entradas sequenciais, enquanto o encoder-decoder (à direita) utiliza transformadas wavelet e operações causais para capturar dependências espaciais e temporais nos dados.

> O bloco Haar Wavelet3D realiza o processo mostrado na visualização abaixo para um grupo de 4 valores em cada dimensão:

> ![3d_wavelet_decomposition](../images/cosmos_tokenizer/3d_wavelet_decomposition.png)
>
> **_ResBlock3D_** aplica a convolução (2 + 1)D com conexões residuais.

> O bloco **_Inverse Haar Wavelet3D_** nada mais é do que a inversão da transformada original, que pega os coeficientes wavelet e reconstrói a imagem (ou vídeo) original.

> O encoder e o decoder são separados do restante da arquitetura do modelo.

Utilizamos a formulação convencional de autoencoder (AE) para modelar o espaço latente do tokenizer contínuo. Para tokenizers discretos, adotamos o Finite-Scalar-Quantization (FSQ) como quantizador do espaço latente. A dimensão latente para os tokenizers contínuos é 16, enquanto para os tokenizers discretos é 6, representando o número de níveis FSQ, que são $(8,8,8,5,5,5)$. Essa configuração corresponde a um vocabulário de tamanho $64.000$.

> O tokenizer contínuo utiliza uma arquitetura Autoencoder, onde uma rede neural comprime os dados de entrada para uma representação latente e depois reconstrói a entrada a partir dessa forma comprimida. A dimensão do espaço latente sendo $16$ significa que cada token é representado por um vetor contínuo de 16 dimensões.

> O tokenizer discreto utiliza Finite-Scalar-Quantization, que mapeia valores contínuos em um conjunto finito de níveis discretos, atribuindo a cada ponto no espaço latente um índice discreto. A dimensão latente no tokenizer discreto ainda é $6$, mas cada dimensão representa mais de um valor. Neste caso, as primeiras $3$ dimensões podem assumir $8$ valores possíveis, e as últimas três podem assumir $5$ valores cada, totalizando $8^3\times 5^3 = 64.000$ possíveis tokens discretos.

---

## English

### Overview

Tokenizers are fundamental building blocks of modern large-scale models. They transform raw data into more efficient representations by learning a bottle-necked latent space discovered in an unsupervised manner. Specifically, visual tokenizers map raw and redundant visual data into compact semantic tokens, making them crucial for handling high-dimensional visual data.

> Mapping the raw data (in the format of pixel values) to a "bottle-necked latent space", means that the original image that is very high-dimensional (for a small RGB image of dimensions $224\times 224 \times 3$ you have a total of $196,608$ features per image), will be compressed into a more useful, smaller form by learning an internal compressed representation (latent space).

> All that to say that the images will be compressed to a lower dimensional form (tokens) when passing through a model _"tokenizer"_ that is trained in a unsupervised manner.

The image bellow illustrates the tokenization training pipeline, where the goal is to train the encoder and decoder, so that the bottleneck token representation maximally preserves visual information in the input.

![Tokenization Pipeline](../images/cosmos_tokenizer/tokenization_pipeline.png)

In the pipeline, an input video is encoded into tokens, which are usually much more compact than the input video. The decoder then reconstructs the input video from the tokens. _Tokenizer training is about learning the encoder and decoder to maximally preserve the visual information in the tokens_.

Tokenizers come in two types: continuous and discrete. Continuous tokenizers encode visual data into continuous latent embeddings, as in latent diffusion models like _Stable Diffusion_ or _VideoLDM_. These embeddings are suitable for models that generate data by sampling from continuous distributions. Discrete tokenizers encode visual data into discrete latent codes, mapping them into quantized indices, as seen in autoregressive transformers such as VideoPoet. This discrete representation is necessary for models such as GPT that are trained with the _cross-entropy loss_.

> _**Continuous tokenizers**_: encodes the data into a continuous, high-dimensional vector space. They are used in diffusion models since these models generate data by sampling from continuous distributions. These embeddings allow the model to interpolate and re-sample variations in the underlying data.

> _**Discrete tokenizers**_: encode the data into discrete latent codes, which are quantized or mapped to a set of distinct finite indices. These tokenizers are often used with _autoregressive models_ that generate sequences one token at a time. The paper cites _GPT models_ and how they're trained with cross-entropy loss, this requires discrete tokens because they treat the generation process as prediction over a fixed vocabulary, and due to the nature of this loss function measuring the difference between predicted and true categorical distributions.

> The main difference between the two tokenizers is how discrete tokenizers map image values to discrete values ($\mathbb{N}$), whereas continuous tokenizers map values to real ($\mathbb{R}$) values, allowing for a higher number of values in the latent space (e.g. _"[...]high-dimensional vector space[...]"_).

> Diffusion models learn from reversing gradual "noising" in real data. The gradual process is why diffusion models need real ($\mathbb{R}$) valued tokens.

The success of tokenizers largely relies on their ability to deliver high compression rates without compromising their subsequent visual reconstruction quality. On one hand, high compression reduces storage and computational demands. On the other hand, excessive compression can lead to the loss of essential visual details. This trade-off presents a significant challenge in tokenizer design.

The following image illustrates the two types of tokens:

![Visualization of continuous and discrete tokenizers](../images/cosmos_tokenizer/token_types.png)

Tokens along spatial ($\frac{H}{S_{HW}} \times \frac{W}{S_{HW}}$) and temporal ($1 + \frac{T}{S_T}$) dimensions, with a spatial compression factor of $S_{HW}$ and a temporal compression factor of $S_T$. The first temporal token represents the first input frame, enabling joint image ($T=0$) and video ($T>0$) tokenization in a shared latent space.

> $S_{HW}$ is the _spatial compression factor_ used to compress the spatial dimensions of an image. This is a key step in the spatial tokenization process, where the input frame is divided into smaller patches or blocks with each one of these being represented by one or more tokens.

> If the original image's dimension is $224\times 224 \times 3$, and the _spatial compression factor_ was 16, the token grid would be $14\times 14$, and each token would hold information on a patch of $16\times 16$ pixels.

> $S_T$ is the _temporal compression factor_, and its used to reduce the number of tokens representing the temporal axis by grouping frames. This is applied in the _Temporal Tokenization_ process, for representation of the number of frames. The addition of $1$ allows the model to treat the initial frame as a special token to support joint image and video tokenization. If the process applied to an image, $T=0$ and the temporal dimension reduces to $1$.

> If the image mentioned above was part of a video with $32$ frames, and the _temporal compression factor_ had a value of $4$, the tokenization process will produce 9 temporal tokens. $1$ for the first frame (adaptability for handling images), and 8 other tokens each compressing 4 frames.

The following table illustrates different visual Tokenizers and their capabilities:

![Different tokenizers and capabilities](../images/cosmos_tokenizer/tokenizers_table.png)

The _Cosmos Tokenizer_ uses a lightweight and computationally efficient architecture with a temporally causal mechanism. Specifically, it employs causal temporal convolution layers and causal temporal attention layers to preserve the natural temporal order of video frames.

> The term "causal" implies that any predictions on a particular frame or time step are based only on that frame and all previous frames, not on any future ones. Therefore "_Causal Temporal Convolution_" means that the feature generation for a given frame only uses data from frame $t$ and earlier.

> The same idea applies to "_Causal Temporal Attention_", where the tokenizer dynamically weighs which frames to focus on when making decisions about the current frame.

The tokenizers are trained directly on high-resolution images and long-duration videos without limiting the categories or aspect ratios. The Cosmos Tokenizer operates across various aspect ratios. They are temporally length-agnostic during inference, capable of tokenizing beyond the temporal length on which it was trained.

The plots bellow show the comparison in performance between the Cosmos Tokenizer and other ones, and denotes the superior quality even at higher compression rates:

![Tokenizer comparisons](../images/cosmos_tokenizer/tokenizer_comparison.png)

### Architecture

Cosmos Tokenizer is designed as an encoder-decoder architecture. Given an input video $x_{0:T} \in \mathbb{R^{(1 + T) \times H \times W \times 3}}$ with $H,\ W,\ T$ being the height, width, and number of frames, the encoder ($\varepsilon$) tokenizes the inputs into a token video $z_{0:T'} \in \mathbb{R^{(1 + T) \times H \times W \times 3}}$, with a spatial compression factor of $s_{H W} = \frac{H}{H'}=\frac{W}{W'}$ and a temporal compression factor of $S_T = \frac{T}{T'}$. The decoder ($\mathcal{D}$) then reconstructs the input video from these tokens, resulting in the reconstructed video $\hat{x}_{0:T} \in \mathbb{R^{(1 + T) \times H \times W \times 3}}$

$$\hat{x}_{0:T} = \mathcal{D}(\varepsilon(x_{0:T}))$$

> This is an overall view of the architecture, where the encoder encodes an input $x_{0:T}$ to tokens $z_{0:T'}$, and the decoder decodes these tokens and outputs $\hat{x}_{0:T}$.

Our architecture employs a temporally causal design, ensuring that each stage processes only current and past frames. _Our tokenizer operates in the wavelet space, where inputs are first processed by a 2-level wavelet transform_. The wavelet transform maps the input video $x_{0:T}$ in a group-wise manner to downsample the inputs by a factor of four along $x, y,$ and $t$. The groups are formed as: $\lbrace x_0, x_{1:4}, x_{5:8}, ..., x_{(T-3):T}\rbrace \rightarrow \lbrace g_0, g_1, g_2, ..., g_{T/4}\rbrace$. Subsequent encoder stages process the frames in a temporally causal manner as $\lbrace g_0, g_{0:1}. g_{0:2}, ...\rbrace \rightarrow \lbrace \xi_0, \xi_1, \xi_2,...  \rbrace$. Successive encoder stages follow a similar scheme, finally outputting the tokens $z_{0:T'}$.

> The _**wavelet transform**_ is a technique for signal processing at multiple scales and resolutions. It differs from more traditional transforms, such as the _Fourier Transform_ that represents data in terms of fixed-frequency sine and cosine waves, by using short, wave-like oscillations that can be scaled and shifted. This transform will decompose both spatial and temporal changes (across fames), compressing and isolating sharp changes in smooth areas.

> The **_Wavelet space_** is the representation of a signal after it has passed through a wavelet transform. So the sentence "_[...]our tokenizer operates in the wavelet space, where inputs are first processed by a 2-level wavelet transform[...]_", means that each spatial and temporal dimension is decomposed, extracting both low-frequency, global, and high-frequency information

> In the process of passing through the 2-level wavelet transform, the data is downsized in each dimension ($x, y, t$) by a factor of $4$. So every group of $4$ pixels ($x_{t:(t+3)}$) in the $3$ dimensions of the image is represented by a compressed group ($g_i$).

> The final tokenizer uses the _Haar Wavelet_, which is one of the simplest wavelet function.

> ![haar_wavelet](../images/cosmos_tokenizer/haar_wavelet.png)

> Wavelet Transforms compress images through decomposition, firstly with a low resolution approximation of the original image, followed by processing vertical, horizontal, and diagonal details in a manner close to the one shown bellow:

> ![Wavelet decomposition](../images/cosmos_tokenizer/wavelet_decomposition.png)

> Resulting in compressed images like this:

> ![Wavelet compressed image](../images/cosmos_tokenizer/wavelet_compressed_image.png)

> And a 2-level wavelet transform would look something like the following:

> ![2-level wavelet](../images/cosmos_tokenizer/2-level_wavelet.png)

The causal design helps adapt models built on top of the tokenizer to downstream Physical AI applications that often operate on the temporal causal setting. the wavelet transform allows us to operate on a more compact video representation that eliminates redundancies in pixel information, allowing the remaining layers to focus on more semantic compression.

Our encoder stages are implemented using a series of residual blocks interleaved with downsampling blocks. In each block, we employ a spatio-temporal factorized 3D convolution, where we first apply a 2D convolution with a kernel size of $1\times k\times k$ to capture spatial information, followed by a temporal convolution with a kernel size of $k\times 1\times 1$ to capture temporal dynamics. We use left padding of k-1 to ensure causality.

> Employs (2 + 1)D convolution.

To capture long-range dependencies, we utilize a spatio-temporal factorized causal self-attention with a global support region. We use the Swish activation function for non-linearity. We leverage Layer Normalization (LayerNorm) instead of Group Normalization (GroupNorm), which prevents large magnitudes from appearing in specific regions of the latent space or reconstructed outputs. The decoder mirrors the encoder replacing the downsampling blocks with an upsampling block. The image bellow depicts an overview of the overall Cosmos Tokenizer architecture.

> Global support region for non linearity means that, tokens interact with all other tokens available at a given time (due to causal architecture restrictions)

> The **_Swish activation function_**, defined by $\operatorname{Swish}^{\beta}(x) = x \cdot sigmoid(\beta x) = \frac{x}{1+e^{-\beta x}}$

> ![Swish activation function](../images/cosmos_tokenizer/swish_activation_function.png)

![Tokenizer architecture](../images/cosmos_tokenizer/tokenizer_architecture.png)

The image depicts the **Overall Cosmos Tokenizer architecture illustrating the integration of temporal causality and an encoder-decoder structure.** Temporal causality (left) processes sequential inputs, while the encoder-decoder (right) leverages wavelet transforms and causal operations to capture spatial and temporal dependencies in the data.

> The **_Haar Wavelet3D_** block does the process shown in the visualization bellow to a group of 4 values in each dimension:

> ![3d_wavelet_decomposition](../images/cosmos_tokenizer/3d_wavelet_decomposition.png)

> The **_ResBlock3D_** applies the (2 + 1)D Convolution with residual connections.

> The **_Inverse Haar Wavelet3D_** is simply the inversion of the original transform, that takes the wavelet coefficients and reconstructs the original image (or video).

> The encoder and decoder are separated by the rest of the model architecture.

We employ the vanilla autoencoder (AE) formulation to model the continuous tokenizer's latent space. For discrete tokenizers, we adopt the Finite-Scalar-Quantization (FSQ) as the latent space quantizer. The latent dimension for the continuous tokenizers is 16, whereas for the discrete tokenizers, it is 6, which represents the number of the FSQ levels, which are $(8,8,8,5,5,5)$. This configuration corresponds to a vocabulary size of $64,000$.

> The continuous tokenizer uses an Autoencoder architecture, where a neural network compresses the input data into a latent representation and then reconstruct the input from this compressed form. The latent space dimension being set by $16$ means each token is represented by a $16$-dimensional continuous vector.

> The discrete tokenizer uses Finite-Scalar-Quantization, that maps continuous values onto a finite set of discrete levels, assigning each point in the latent space to a discrete index. The latent dimension in the discrete tokenizer is still $6$, but each dimension represents more than one value. In this case, the first $3$ dimensions can take $8$ possible values, and the last three can each take $5$ possible values for a total of $8^3\times 5^3 = 64,000$ possible discrete tokens.

---

## Referências

- [Cosmos World Foundation Model Platform for Physical AI](https://arxiv.org/abs/2501.03575)

- [What is Wavelet Transform?Fourier vs Wavelet Transform|CWT-DWT|Wavelet Transform in Image Processing](https://www.youtube.com/watch?v=pUty-98Km_0)

- [Discrete tools for virtual sculpture - Scientific Figure on ResearchGate. Available from: https://www.researchgate.net/figure/D-Haar-wavelet-decomposition_fig1_220868824 [accessed 24 Jul 2025]](https://www.researchgate.net/figure/D-Haar-wavelet-decomposition_fig1_220868824)
