[![license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# Coronagraph Inpainting - Deep Learning approach
The coronagraph data calibration routine in [SolarSoft](https://www.lmsal.com/solarsoft/) employs a fuzzy algorithm to generate a level 1 FITS file, addressing missing data holes. This project endeavors to introduce a novel deep learning methodology, leveraging [Partial Convolutions](https://arxiv.org/abs/1804.07723), to effectively address typical manifestations of data loss in coronagraphic imagery. By harnessing the capabilities of [Partial Convolutions](https://arxiv.org/abs/1804.07723), this approach aims to enhance the robustness and efficiency of data restoration processes.\
## Approach
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

### New proposal:
Even though this approach achieves state-of-the-art performance, larger information loss chunks are not very well inpainted. To address this for our application, we'll make a network trainable by parts. This is, constantly going back to the number of input channels and evaluating the loss between the original image masked with the last mask and the output of the network. This adaptive loss approach ensures that the missing holes are being filled from the edges to the center.

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
- [Email](jorged.encyso@gmail.com)
- [GitHub](https://github.com/Jorgedavyd)