# Implementation in pytorch of Auto-Encoders: Linear, Denoising, Contractive Auto-encoders

### 1. Data Loading
MNIST dataset was loading with batch size of 64 and intermediates operations like Normalisation () 

## I. Linear Auto-Encoders
In this case of Deep linear autoencoder, when we use six hidden layers with the following linear structure inside the encoder: 28 * 28 -> 512 -> 256 -> 128 -> 64 -> 16 -> 3, the performance of the model is very poor because the size of the latent code is reduced, Network architecture is too deep Compare to the Simple Linear Autoencoder model which performs better despite its simple architecture: 28 * 28 -> 128 -> 64 -> 32 -> 16 -> 8 

**Result:** 

![Linear_auto_encoder](./assets/Linear_auto_encoder.png)
Deep Linear auto-encoder reconstruction of eigth MNIST images

![Linear_auto_encoder](./assets/simple_linear.png)
Simple Linear auto-encoder reconstruction of eigth MNIST images

## II. Denoising Auto-Encoders


The detailed architecture of the denoising autoencoder is outlined as:

### Encoder
The encoder compresses the input image into a lower-dimensional representation using the following layers:
- **Input:** 1 channel (28 x 28)
- **Convolutional Layer:** 1 -> 16, kernel size: 3x3, stride: 2, padding: 1, activation: ReLU
- **Convolutional Layer:** 16 -> 32, kernel size: 3x3, stride: 2, padding: 1, activation: ReLU

### Decoder
The decoder reconstructs the original image from the compressed representation using the following layers:
- **Transposed Convolutional Layer:** 32 -> 16, kernel size: 2x2, stride: 2, activation: ReLU
- **Transposed Convolutional Layer:** 16 -> 1, kernel size: 2x2, stride: 2, activation: Sigmoid (The Sigmoid activation function scales the output pixel values to the range [0, 1]).
- **Output:** 1 channel (28 x 28)
- **Result:** 
![Denoising_auto_encoder](./assets/denoising_0.8_out.png)
Denoising auto-encoder reconstruction of six MNIST images

## III. Contractive Auto-Encoders
For the contractive autoencoder, we use as architecture of the encoder two fully connected layers with relu as activation function and for the decoder, we use also two layers where the first layer has relu as activation function and the second has sigmoid as activation function.
The loss function of a Contractive Autoencoder typically consists of two main components: the reconstruction loss and the contractive penalty.
It's written as follow:

$$L = MSE(x, \hat{x}) + \lambda * contractive \  penalty$$

where $$contractive \ penalty = \Big\Vert\sum_{i} \frac{\partial h_i}{\partial x}\Big\Vert_F^2$$

$\lambda$ is the hyperparameter that controls the relative importance of the contractive penalty term compared to the reconstruction loss.

We use two values of lambda two train our model: $\lambda = 5$ and $\lambda = 1e-4$.

For $\lambda = 5$, the model is very sensitive to the variations in the input data. We add a gaussian noise to the data then the model is not able to reconstruct well the data at some point.

For $\lambda = 1e-4$, our model perform well on unseen data. It's not too sensitive to the variation.




**Result:** 

![Contractive_auto_encoder](./assets/Img_lambda=5.png)
Contractive auto-encoder reconstruction of eigth MNIST images for lambda = 5

![Contractive_auto_encoder](./assets/Img_lambda_very_small.png)
Contractive auto-encoder reconstruction of eigth MNIST images for lambda = 1e-4

### Build With

**Language:** Python

**Package:** torchvision, matplotlib, Pytorch

### Run Locally

Clone the project
```bash
    git clone https://github.com/Omer-alt/Auto-Encoders.git
```

Go to the project directory
```bash
    cd Auto-Encoders
```

To run linear auto-encoder
```bash
    cd Linear_Auto_encode
    python3 linearAutoencoder.py

```

To run denoising auto-encoder
```bash
    cd Denoising-Auto-encoder
    python3 main.py
```

To run contractive auto-encoder
```bash
    cd ContractiveAutoencoder
    python3 ContractiveAutoencoder.py
```

Go to the project directory
### License

[MIT](https://choosealicense.com/licenses/mit/)