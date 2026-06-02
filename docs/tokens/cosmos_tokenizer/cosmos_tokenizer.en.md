
## English

Below we present excerpts from chapter 5 of the article **_Cosmos World Foundation Model Platform for Physical AI_**, interspersed with comments clarifying and/or contextualizing the content of its paragraphs.

---

Passages formatted similarly to this paragraph correspond to translated excerpts from the article.

> Passages formatted similarly to this paragraph are comments regarding the paragraph directly above.

---

### Summary

The _Cosmos Tokenizer_ is composed by $2 $parts, an **encoder** and a **decoder**. The **encoder** starts with a _Haar Wavelet 3D_ operation, to compress the image or video, then a number of blocks composed by a _Causal ResBlock3D_ layer, _Causal DownSample3D_ layers, and a _Causal SpatioTemporalAttn_ layer.
The **decoder** mirrors this architecture, substituting the downsample layers for _Causal UpSample3D_ layers, and in the end substituting the _Haar Wavelet3D_ for its inverse.

Both parts are trained together, with supervision only in the output of the decoder. These tokenizers work for both autoregressive and diffusion architectures, being able to tokenize images/videos in a discrete (for AR models), or continuous manner (for diffusion models).

The _Cosmos Tokenizer_ reaches higher performance in less time than other tokenizers, mostly due to its architecture, while also being able to process multiple types of compression rates, and work ubiquitously for images and videos.

### Overview

Tokenizers are fundamental building blocks of modern large-scale models. They transform raw data into more efficient representations by learning a bottle-necked latent space discovered in an unsupervised manner. Specifically, visual tokenizers map raw and redundant visual data into compact semantic tokens, making them crucial for handling high-dimensional visual data.

> Mapping the raw data (in the format of pixel values) to a "bottle-necked latent space", means that the original image that is very high-dimensional (for a small RGB image of dimensions $224\times 224 \times 3$ you have a total of $196,608$ features per image), will be compressed into a more useful, smaller form by learning an internal compressed representation (latent space).
>
> All that to say that the images will be compressed to a lower dimensional form (tokens) when passing through a model _"tokenizer"_ that is trained in a unsupervised manner.

The image bellow illustrates the tokenization training pipeline, where the goal is to train the encoder and decoder, so that the bottleneck token representation maximally preserves visual information in the input.

![Tokenization Pipeline](../images/cosmos_tokenizer/tokenization_pipeline.png)
_"Figura S2.1-F6-ENG — Tokenization Pipelines"_

In the pipeline, an input video is encoded into tokens, which are usually much more compact than the input video. The decoder then reconstructs the input video from the tokens. _Tokenizer training is about learning the encoder and decoder to maximally preserve the visual information in the tokens_.

Tokenizers come in two types: continuous and discrete. Continuous tokenizers encode visual data into continuous latent embeddings, as in latent diffusion models like _Stable Diffusion_ or _VideoLDM_. These embeddings are suitable for models that generate data by sampling from continuous distributions.
Discrete tokenizers encode visual data into discrete latent codes, mapping them into quantized indices, as seen in autoregressive transformers such as VideoPoet. This discrete representation is necessary for models such as GPT that are trained with the _cross-entropy loss_.

> _**Continuous tokenizers**_: encodes the data into a continuous, high-dimensional vector space. They are used in diffusion models since these models generate data by sampling from continuous distributions. These embeddings allow the model to interpolate and re-sample variations in the underlying data.
>
> _**Discrete tokenizers**_: encode the data into discrete latent codes, which are quantized or mapped to a set of distinct finite indices. These tokenizers are often used with _autoregressive models_ that generate sequences one token at a time.
> The paper cites _GPT models_ and how they're trained with cross-entropy loss, this requires discrete tokens because they treat the generation process as prediction over a fixed vocabulary, and due to the nature of this loss function measuring the difference between predicted and true categorical distributions.
>
> The main difference between the two tokenizers is how discrete tokenizers map image values to discrete values ($\mathbb{N}$), whereas continuous tokenizers map values to real ($\mathbb{R}$) values, allowing for a higher number of values in the latent space (e.g. _"[...]high-dimensional vector space[...]"_).
>
> Diffusion models learn from reversing gradual "noising" in real data. The gradual process is why diffusion models need real ($\mathbb{R}$) valued tokens.

The success of tokenizers largely relies on their ability to deliver high compression rates without compromising their subsequent visual reconstruction quality. On one hand, high compression reduces storage and computational demands. On the other hand, excessive compression can lead to the loss of essential visual details. This trade-off presents a significant challenge in tokenizer design.

The following image illustrates the two types of tokens:

![Visualization of continuous and discrete tokenizers](../images/cosmos_tokenizer/token_types.png)
_"Figura S2.2-F7-ENG — Visualization of discrete Tokenizers"_
Tokens along spatial ($\frac{H}{S_{HW}} \times \frac{W}{S_{HW}}$) and temporal ($1 + \frac{T}{S_T}$) dimensions, with a spatial compression factor of $S_{HW}$ and a temporal compression factor of $S_T$. The first temporal token represents the first input frame, enabling joint image ($T=0$) and video ($T>0$) tokenization in a shared latent space.

> $S_{HW}$ is the **_spatial compression factor_** used to compress the spatial dimensions of an image. This is a key step in the spatial tokenization process, where the input frame is divided into smaller patches or blocks with each one of these being represented by one or more tokens.
>
> If the original image's dimension is $224\times 224 \times 3$, and the _spatial compression factor_ was 16, the token grid would be $14\times 14$, and each token would hold information on a patch of $16\times 16$ pixels.
>
> $S_T$ is the _temporal compression factor_, and its used to reduce the number of tokens representing the temporal axis by grouping frames. This is applied in the _Temporal Tokenization_ process, for representation of the number of frames.
> The addition of $1$ allows the model to treat the initial frame as a special token to support joint image and video tokenization. If the process applied to an image, $T=0$ and the temporal dimension reduces to $1$.
>
> If the image mentioned above was part of a video with $32$ frames, and the _temporal compression factor_ had a value of $4$, the tokenization process will produce 9 temporal tokens. $1$ for the first frame (adaptability for handling images), and 8 other tokens each compressing 4 frames.

The following table illustrates different visual Tokenizers and their capabilities:

![Different tokenizers and capabilities](../images/cosmos_tokenizer/tokenizers_table.png)
_"Figura S2.2-F8-ptbr — Different Tokenizers and its capabilites"_
The _Cosmos Tokenizer_ uses a lightweight and computationally efficient architecture with a temporally causal mechanism. Specifically, it employs causal temporal convolution layers and causal temporal attention layers to preserve the natural temporal order of video frames.

> The term "causal" implies that any predictions on a particular frame or time step are based only on that frame and all previous frames, not on any future ones. Therefore "_Causal Temporal Convolution_" means that the feature generation for a given frame only uses data from frame $t$ and earlier.
>
> The same idea applies to "_Causal Temporal Attention_", where the tokenizer dynamically weighs which frames to focus on when making decisions about the current frame.

The tokenizers are trained directly on high-resolution images and long-duration videos without limiting the categories or aspect ratios. The Cosmos Tokenizer operates across various aspect ratios. They are temporally length-agnostic during inference, capable of tokenizing beyond the temporal length on which it was trained.

The plots bellow show the comparison in performance between the Cosmos Tokenizer and other ones, and denotes the superior quality even at higher compression rates:

![Tokenizer comparisons](../images/cosmos_tokenizer/tokenizer_comparison.png)
_"Figura S2.2-F8-ptbr — Comparisons between Tokenizers"_

### Architecture

Cosmos Tokenizer is designed as an encoder-decoder architecture.
Given an input video $x_{0:T}\in\mathbb{R^{(1 + T)\times H\times W\times 3}}$ with $H,\ W,\ T$ being the height, width, and number of frames, the encoder ($\varepsilon$) tokenizes the inputs into a token video $z_{0:T'}\in\mathbb{R^{(1+T) \times H\times W\times 3}}$, with a spatial compression factor of $s_{HW} = \frac{H}{H'}=\frac{W}{W'}$ and a temporal compression factor of $S_T = \frac{T}{T'}$.
The decoder ($\mathcal{D}$) then reconstructs the input video from these tokens, resulting in the reconstructed video $\hat{x}_{0:T} \in \mathbb{R^{(1 + T) \times H \times W \times 3}}$

$$\hat{x}_{0:T} = \mathcal{D}(\varepsilon(x_{0:T}))$$

> This is an overall view of the architecture, where the encoder encodes an input $x_{0:T}$ to tokens $z_{0:T'}$, and the decoder decodes these tokens and outputs $\hat{x}_{0:T}$.

Our architecture employs a temporally causal design, ensuring that each stage processes only current and past frames. _Our tokenizer operates in the wavelet space, where inputs are first processed by a 2-level wavelet transform_.
The wavelet transform maps the input video $x_{0:T}$ in a group-wise manner to downsample the inputs by a factor of four along $x, y,$ and $t$.
The groups are formed as: $\lbrace x_0,x_{1:4},x_{5:8},...,x_{(T-3):T}\rbrace\rightarrow\lbrace g_0,g_1,g_2,..., g_{T/4}\rbrace$. Subsequent encoder stages process the frames in a temporally causal manner as $\lbrace g_0, g_{0:1}. g_{0:2}, ...\rbrace \rightarrow \lbrace \xi_0, \xi_1, \xi_2,...  \rbrace$. Successive encoder stages follow a similar scheme, finally outputting the tokens $z_{0:T'}$.

> The _**wavelet transform**_ is a technique for signal processing at multiple scales and resolutions. It differs from more traditional transforms, such as the _Fourier Transform_ that represents data in terms of fixed-frequency sine and cosine waves, by using short, wave-like oscillations that can be scaled and shifted.
> This transform will decompose both spatial and temporal changes (across fames), compressing and isolating sharp changes in smooth areas.
>
> The **_Wavelet space_** is the representation of a signal after it has passed through a wavelet transform. So the sentence "_[...]our tokenizer operates in the wavelet space, where inputs are first processed by a 2-level wavelet transform[...]_", means that each spatial and temporal dimension is decomposed, extracting both low-frequency, global, and high-frequency information
>
> In the process of passing through the 2-level wavelet transform, the data is downsized in each dimension ($x, y, t$) by a factor of $4$. So every group of $4$ pixels ($x_{t:(t+3)}$) in the $3$ dimensions of the image is represented by a compressed group ($g_i$).
>
> The final tokenizer uses the _Haar Wavelet_, which is one of the simplest wavelet function.
>
> ![haar_wavelet](../images/cosmos_tokenizer/haar_wavelet.png)
>
> Wavelet Transforms compress images through decomposition, firstly with a low resolution approximation of the original image, followed by processing vertical, horizontal, and diagonal details in a manner close to the one shown bellow:
>
> ![Wavelet decomposition](../images/cosmos_tokenizer/wavelet_decomposition.png)
>
> Resulting in compressed images like this:
>
> ![Wavelet compressed image](../images/cosmos_tokenizer/wavelet_compressed_image.png)
>
> And a 2-level wavelet transform would look something like the following:
>
> ![2-level wavelet](../images/cosmos_tokenizer/2-level_wavelet.png)

The causal design helps adapt models built on top of the tokenizer to downstream Physical AI applications that often operate on the temporal causal setting. the wavelet transform allows us to operate on a more compact video representation that eliminates redundancies in pixel information, allowing the remaining layers to focus on more semantic compression.

Our encoder stages are implemented using a series of residual blocks interleaved with downsampling blocks.
In each block, we employ a spatio-temporal factorized 3D convolution, where we first apply a 2D convolution with a kernel size of $1\times k\times k$ to capture spatial information, followed by a temporal convolution with a kernel size of $k\times 1\times 1$ to capture temporal dynamics. We use left padding of k-1 to ensure causality.

> Employs (2 + 1)D convolution.

To capture long-range dependencies, we utilize a spatio-temporal factorized causal self-attention with a global support region. We use the Swish activation function for non-linearity. We leverage Layer Normalization (LayerNorm) instead of Group Normalization (GroupNorm), which prevents large magnitudes from appearing in specific regions of the latent space or reconstructed outputs.
The decoder mirrors the encoder replacing the downsampling blocks with an upsampling block. The image bellow depicts an overview of the overall Cosmos Tokenizer architecture.

> Global support region for non linearity means that, tokens interact with all other tokens available at a given time (due to causal architecture restrictions)
>
> The **_Swish activation function_**, defined by $\operatorname{Swish}^{\beta}(x) = x \cdot sigmoid(\beta x) = \frac{x}{1+e^{-\beta x}}$
>
> ![Swish activation function](../images/cosmos_tokenizer/swish_activation_function.png)

![Tokenizer architecture](../images/cosmos_tokenizer/tokenizer_architecture.png)
_"Figura S2.2-F8-ptbr — Tokenizers architectures"_

The image depicts the **Overall Cosmos Tokenizer architecture illustrating the integration of temporal causality and an encoder-decoder structure.** Temporal causality (left) processes sequential inputs, while the encoder-decoder (right) leverages wavelet transforms and causal operations to capture spatial and temporal dependencies in the data.

> The **_Haar Wavelet3D_** block does the process shown in the visualization bellow to a group of 4 values in each dimension:
>
> ![3d_wavelet_decomposition](../images/cosmos_tokenizer/3d_wavelet_decomposition.png)
>
> Both the **_ResBlock3D_** and the **_DownSample3D_** apply (2 + 1)D Convolutions with the difference being the skip connections in the **_ResBlock3d_**.
>
> The **_Inverse Haar Wavelet3D_** is simply the inversion of the original transform, that takes the wavelet coefficients and reconstructs the original image (or video).
>
> The encoder and decoder are separated by the rest of the model architecture.

We employ the vanilla autoencoder (AE) formulation to model the continuous tokenizer's latent space. For discrete tokenizers, we adopt the Finite-Scalar-Quantization (FSQ) as the latent space quantizer.
The latent dimension for the continuous tokenizers is 16, whereas for the discrete tokenizers, it is 6, which represents the number of the FSQ levels, which are $(8,8,8,5,5,5)$. This configuration corresponds to a vocabulary size of $64,000$.

> The continuous tokenizer uses an Autoencoder architecture, where a neural network compresses the input data into a latent representation and then reconstruct the input from this compressed form. The latent space dimension being set by $16$ means each token is represented by a $16$-dimensional continuous vector.
>
> The discrete tokenizer uses Finite-Scalar-Quantization, that maps continuous values onto a finite set of discrete levels, assigning each point in the latent space to a discrete index.
> The latent dimension in the discrete tokenizer is still $6$, but each dimension represents more than one value. In this case, the first $3$ dimensions can take $8$ possible values, and the last three can each take $5$ possible values for a total of $8^3\times 5^3 = 64,000$ possible discrete tokens.

### Training Strategy

We employ a joint training strategy by alternating mini-batches of images and videos at a preset frequency. We only supervise the final output of our tokenizer's decoder. We do not use auxiliary losses tapped into the latent spaces.

> "_[...] mini-batches of images and videos at a preset frequency [...]_", means that the model uses batches of both images and videos during training, in an alternating manner, set at a preset frequency, or in other words, switching between the two every $N$ number of batches.
>
> The idea of joint training for both images and videos, gets the network exposed to both single-framed and multi-framed data, so its latent space becomes more suitable for both types of input data.
>
> Both the discrete and the continuous tokenizers, map continuous data to a latent space, and the sentence "_[...] We do not use auxiliary losses tapped into the latent spaces [...]_" means that the training does not use any additional losses in order to encourage certain behaviours or properties (such as disentanglement, compactness, or interpretability) in that latent space.

We employ a two-stage training scheme. In the first stage, we optimize with the _L1 Loss_ that minimizes the pixel-wise RGB difference between the input and reconstructed video ($\hat{x}_{0:T}$) given by:

$$\mathcal{L}_1 = ||\hat{x}_{0:T} - x_{0:T}||_1$$

> $\mathcal{L}_1$ loss is another name for **_Mean Absolute Error_**, where instead of squaring the difference between the predicted value and its actual value, in order to make the values positive (as performed in **_Mean Squared Error_**, or $\mathcal{L}_2$ loss), we take the absolute difference between the two values.
>
> The function is represented in the [**_Einstein Notation_**](https://en.wikipedia.org/wiki/Einstein_notation).

And the perceptual loss based on the VGG-19 features, given by:

$$\frac{1}{L}\sum_{l=1}^{L} \sum_{t}^{}{ \alpha_l || \mathrm{VGG}_l(\hat{x}_t) - \mathrm{VGG}_l(x_t) ||1}$$

Where $\mathrm{VGG}_l(\cdot) \in \mathbb{R}^{H\times W\times C}$ is the features from the $l$-th layer of a pre-trained **_VGG-19_** network, $L$ is the number of layers considered, and $\alpha_l$ is the weight of the $l$-th layer.

> A perceptual loss is a way of analysing how well an image is being reconstructed, generated, or enhanced. It compares the feature representations of images rather than the pixel-wise difference. So where a simple $\mathcal{L}_1$ loss measures the pixel difference between two images at a given step, the perceptual loss above, will measure how "far apart" two feature maps are from one another.
>
> The loss function above, determines how different the feature map (compute the $\mathcal{L}_1$ loss), at a given layer $l$ of the VGG-19 model, between the real image and the generated image at the layer $l$. After measuring the absolute differences, it multiplies that value by the weight of that layer ($\alpha_l$).

In the second stage, we use the optical flow ($\mathrm{OF}$) loss to handle the temporal smoothness of reconstructed videos,

$$\frac{1}{T}\sum_{t=1}^{T}||\mathrm{OF}(\hat{x}_{t}, \hat{x}_{t - 1}) - \mathrm{OF}({x}_{t}, {x}_{t - 1})||_1 + \frac{1}{T}\sum_{t=0}^{T - 1}||\mathrm{OF}(\hat{x}_{t}, \hat{x}_{t - 1}) - \mathrm{OF}({x}_{t}, {x}_{t - 1})||_1$$

> **_Optical Flow_** is the apparent motion of objects, surfaces and edges between consecutive frames in a video sequence. It is a vector field where each vector represents the motion of pixel from one frame to the next. This process, helps understand how and where things move in a scene.
>
> This loss is used to encourage a video model to preserve the motion patterns present in the original video. $\mathrm{OF}(\hat{x}_{t}, \hat{x}_{t - 1})$ is the optical flow between reconstructed frames $\hat{x}_{t}, \hat{x}_{t - 1}$, and $\mathrm{OF}(x_{t}, x_{t - 1})$ is the optical flow between the actual frames $x_{t}, x_{t - 1}$.
>
> The function for $\mathcal{L}_{Flow}$ sums the $\mathcal{L}_1$ loss between the frames from $(t=1 \rightarrow t=T)$, and the loss between the frames $(t=0 \rightarrow t=T-1)$. This is done in order to penalise mismatches between reconstructed and original video frames over all consecutive frame pairs.

Additionally, we use adversarial loss in the fine-tuning stage to further enhance reconstruction details, particularly at large compression rates.

> Adversarial loss is a technique used in _GANs_ where a discriminator network tries to distinguish between real and generated images.

We train the image tokenizers (CI and DI) at two compression rates: $8\times 8$ and $16\times 16$. Similarly we train the video tokenizers (CV and DV) at three compression rates: $4\times 8\times 8$, $8\times 8\times 8$, and $8\times 16\times 16$.
Here, the compression rates are expressed as $H\times W$ for images and $T\times H\times W$ for videos, where $T$ represents the temporal dimension and $H$ and $W$ represent the spatial dimensions.

> The compression rates determine how much of the input's resolution is reduced during tokenization.

For the video tokenizers, we create two variants:

1. **Cosmos-0.1-Tokenizer**: Trained using mini-batches sampling a smaller number of video frames ($49$ frames for CV and $17$ frames for DV).
2. **Cosmos-1.0-Tokenizer**: Trained using mini-batches sampling a larger number of video frames ($121$ frames for CV and $49$ frames for DV).

This approach ensures flexibility in handling varying temporal and spatial resolutions for image and video data.

### Results

![Tokenizer Evaluation 1](../images/cosmos_tokenizer/tokenizer_evaluation_1.png)

![Tokenizer Evaluation 2](../images/cosmos_tokenizer/tokenizer_evaluation_2.png)

We evaluate our Cosmos Tokenizer suite on various image and video benchmark datasets. For the evaluation of image tokenizers, we follow prior art to evaluate **MS-COCO 2017** and **ImageNet-1K**. We use the **MS-COCO 2017** validation subset of $5,000$ images, and **ImageNet-1K** validation subset of $50,000$ images as image evaluation benchmark.

**TokenBench**. For video tokenizer evaluation, there is not yet a standard benchmark for high-resolution and long-duration videos.
To this end, we introduce a benchmark called _TokenBench_ to cover a wide variety of domains, including robotic manipulation, driving, egocentric, and web videos, and standardize the evaluation. We resort to existing video datasets that are commonly used for various tasks, including **BDD100K**, **EgoExo-4D**, **BridgeData V2**, and **Panda-70M**.
We randomly sample $100$ videos from each dataset and preprocess them by taking the first $10$ seconds and resizing the short size to $1080$. For **Panda-70M**, we manually filter out the videos with low-quality content and small motions. For **EgoExo-4D** , we randomly pick $100$ scenes and sample one egocentric video and one exocentric video. This results in a total of $500$ videos.

> **Egocentric** images are from first-person viewpoint, whereas **Exocentric** images are from third-person viewpoints.

In addition to _TokenBench_, we also evaluate our video tokenizers on the **DAVIS** dataset at $1080p$ resolution.

**Baselines and evaluation metrics**. We evaluate our tokenizers at various compression rates to showcase their effectiveness for different computational needs.
We compare each of these tokenizers with state-of-the-art image and video tokenizers. The evaluation metrics include **_Peak Signal-to-Noise Ratio (PSNR)_**, **_Structural Similarity(SSIM)_**, **_reconstruction Fréchet Inception Distance (rFID)_** for images and **_reconstruction Fréchet Video Distance (rFVD)_** for videos.

> **_Peak Signal-to-Noise Ratio (PSNR)_**: Measure the average difference between the original and reconstructed images/videos with focus on pixel-level fidelity. Higher **PSNR** values imply better quality and less distortion (not necessarily for human vision).
> $$PSNR = 10 \cdot \log_{10} (\frac{{MAX}_I^2}{MSE})$$
> Where $MAX$ is the maximum possible pixel value of the image ($255$ for $8$ bit images).
>
> **_Structural Similarity Index Measure(SSIM)_**: Measures the perceived structural similarity by comparing luminance, contrast and structure. This metric is better aligned with human vision, compared to $PSNR$.
> $$SSIM(x, \hat{x}) = \frac{(2\mu_x\mu_{\hat{x}} + c_1)(2\sigma_{x\hat{x}} + c_2)}{(\mu_x^2 + \mu_{\hat{x}}^2 + c_1)(\sigma_x^2 + \sigma_{\hat{x}}^2 + c_1)}$$
> Where:
>
> - $\mu_x, \mu_{\hat{x}}$ are means of original and reconstructed patches.
> - $\sigma_x^2, \sigma_{\hat{x}}^2$ are variances of the patches.
> - $\sigma_{x\hat{x}}$ is the covariance between patches.
> - $c_1, c_2$ are small constants for division stabilization.
>
> **_reconstruction Fréchet Inception Distance (rFID)_**: Measures distributional similarity between the abstract features of the original and reconstructed images. Lower values indicate reconstructions are more statistically similar to real images in high-level feature spaces.
> $$rFID(X,Y) = ||\mu_X - \mu_Y||_2^2 + Tr (\sum X + \sum Y - 2(\sum X\sum Y)^{1/2})$$
>
> - $X,Y$ are collections of features from the real and reconstructed images.
> - $\mu_X, \mu_Y$ are means of original and reconstructed feature vectors.
> - $\sum X, \sum Y$ are covariance matrices.
> - $Tr$ is the matrix trace.
>
> **_reconstruction Fréchet Video Distance (rFVD)_**: Measures how close the distribution of reconstructed videos is to real videos in feature spaces. Lower values indicate not only more realistic looking videos, but motion an temporal dynamic matching the original videos.
> $$rFVD(X,Y) = ||\mu_X - \mu_Y||_2^2 + Tr (\sum X + \sum Y - 2(\sum X\sum Y)^{1/2})$$
>
> - $X,Y$ are collections of features from the real and reconstructed videos.
> - $\mu_X, \mu_Y$ are means of original and reconstructed feature vectors of the videos.
> - $\sum X, \sum Y$ are the video's covariance matrices.
> - $Tr$ is the matrix trace (same as above).

**Quantitative results** As shown in both tables ($5,6$), Cosmos Tokenizer achieves state-of-the-art performance in all the metrics compared to prior arts on both the _DAVIS_ video dataset and _TokenBench_, with a spatial-temporal compression ratio of $4\times 8\times 8$.
Moreover, even with $2\times$ and $8\times$ higher compression ratios, Cosmos Tokenizer is often comparable or even better than prior art at $8\times 8$ compression ratio, as shown in tables $7$, and $8$.

As shown in these tables, compared to prior arts, Cosmos Tokenizer consistently achieves state-of-the-art results with a compression ratio of $8\times 8$. More importantly, at a $4\times$ larger compression ratio of $16\times 16$, the image quality of Cosmos Tokenizer is often comparable or even better than prior art at $8\times 8$ compression ratio.

As shown in table $9$, for both image and video tokenizers, Cosmos Tokenizer is $2\times \ ~ \ 12\times$ faster while maintaining the smallest model size compared to prior arts, showing that Cosmos Tokenizer has high efficiency for encoding and decoding visual content.

---

## Referências | References

- [Cosmos World Foundation Model Platform for Physical AI arXiv:2501.03575](https://arxiv.org/abs/2501.03575)

- [What is Wavelet Transform?Fourier vs Wavelet Transform|CWT-DWT|Wavelet Transform in Image Processing](https://www.youtube.com/watch?v=pUty-98Km_0)

- [Discrete tools for virtual sculpture - Scientific Figure on ResearchGate. Available from: https://www.researchgate.net/figure/D-Haar-wavelet-decomposition_fig1_220868824 [accessed 24 Jul 2025]](https://www.researchgate.net/figure/D-Haar-wavelet-decomposition_fig1_220868824)

- [NVIDIA Lança Plataforma Cosmos World Foundation Model para Acelerar o Desenvolvimento da IA Física](https://blog.nvidia.com.br/blog/nvidia-lanca-plataforma-cosmos-world-foundation-model-para-acelerar-o-desenvolvimento-da-ia-fisica/)

- [Uber Teams Up with NVIDIA to Accelerate Autonomous Mobility](https://investor.uber.com/news-events/news/press-release-details/2025/Uber-Teams-Up-with-NVIDIA-to-Accelerate-Autonomous-Mobility/default.aspx)
