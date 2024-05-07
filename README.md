[![license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![code-style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Coronagraph Inpainting
The coronagraph data calibration routine in [SolarSoft](https://www.lmsal.com/solarsoft/) employs a fuzzy algorithm to generate a level 1 FITS file, addressing missing data holes. This project endeavors to introduce a novel deep learning methodology, leveraging [Partial Convolutions](https://arxiv.org/abs/1804.07723), to effectively address typical manifestations of data loss in coronagraphic imagery. By harnessing the capabilities of [Partial Convolutions](https://arxiv.org/abs/1804.07723), this approach aims to enhance the robustness and efficiency of data restoration processes.\

## Partial Convolutions
To adapt the backbone's architecture for coronagraph inpainting, understanding the nature of information loss is crucial. We deal with squared missing holes of size (32,32), which can compound into larger ones. We aim to enhance partial convolution efforts by comprehensively understanding the constraints of our problem.

### Convolution:

$$
O = W^T (X \odot M) \frac{sum(1)}{sum(M)} + b
$$

### Mask Update:

$$
m' = \begin{cases} 
1 & \text{if } \sum(M) > 0 \\
0 & \text{otherwise}
\end{cases}
$$

## New proposals:

1. **Multi-layered UNet like Partial convolution Network**: Recursive architecture designed to strike progressive locations of the image through every layer. 

2. **Multi-layered Fourier Space Based Variational Autoencoder**: Generative model whose decoder is designed to continuesly update the mask towards the center of each loss data chunk. For faster inference and resource management, Fourier space convolutions are used as decoder and skip connections to the decoder are made through fourier space convolutions.

3. **Multi-layered Fourier Space Based Partial Convolution Network**: Designed a Partial Convolution UNet like network that analyses both linear spaces to extract features. 

1 and 3 are trained in a **Teacher Forcing** fashion, as the default training method for recursive architectures. Also, they have Cross Multihead Attention in the channel dimention between: $Re[\mathcal{F}[X]], Im[\mathcal{F}[X]]$ and $X$ spaces. X being the image space. The skip connection of 3 is by a convolution operation between the output of encoder layers ($\mathcal{F}[X]$) and the new Fourier spaces on the decoder.

## Convolution theorem
There is a lot of relations between the Fourier Space and Convolutions:

$$r(x) = [u * v](x) = \mathcal{F}^{-1}[U \odot V]$$

Given that $ U, V $ are both the Fourier space representations of $u, v$.

## Instalation

These models will be implemented in the Python library [CorKit](https://github.com/Jorgedavyd/corkit).
```bash
pip install corkit
```
as part of the `level_1` routines.

## Acknowledgements

This project was inspired by the work of SolarSoft developers and the researchers behind partial convolutional layers, which form the backbone of this application.

## Contact  

- [Linkedin](https://www.linkedin.com/in/jorge-david-enciso-mart%C3%ADnez-149977265/)
- Email: jorged.encyso@gmail.com
- [GitHub](https://github.com/Jorgedavyd)