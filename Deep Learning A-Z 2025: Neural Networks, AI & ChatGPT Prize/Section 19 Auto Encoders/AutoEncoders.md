- **Auto Encoders** are special type of neural networks that learn to compress data into a compact form and then reconstruct it to closely match the original input. They consists of:
  - Encoder that captures important features by reducing dimentionality
  - Decoder that rebuilts the data from this compressed representation
The model trains by minimizing reconstruction error using loss functions like Mean Squared Error or Binary Cross-Entropy. These are applied in tasks such as noise removal, error detection and feature extraction where capturing efficient data representations is important.

<img width="828" height="546" alt="image" src="https://github.com/user-attachments/assets/33773610-2a83-403c-afb2-238bf0347036" />


Additional Reading: 
[Neural nets are impressively Good at Compression by Malte Skarupke (2016)](https://probablydance.com/2016/04/30/neural-networks-are-impressively-good-at-compression/)

- Biases

- Training of Auto Encoder

<img width="1216" height="653" alt="image" src="https://github.com/user-attachments/assets/4c9a9b63-f4e8-406e-9658-737d207964ea" />

Additional Reading:
[Building Auto Encoders in Keras by Francois Chollet 2016](https://blog.keras.io/building-autoencoders-in-keras.html)

- overcomplete hidden layers
- Sparse AutoEncoders
- Denoising AutoEncoders
- Contractive AutoEncoders
- Stacked AutoEncoders 
- Deep AutoEncoders
