### Variational Autoencoders (VAEs) 

**Concept**  
- VAEs are **generative models** that learn a smooth, probabilistic latent space.  
- Unlike standard autoencoders, they don’t just compress and reconstruct data—they can also **generate new samples**.  
- They balance reconstruction accuracy with latent space regularization using **variational inference**.  

**Architecture**  
1. **Encoder**  
   - Maps input data (e.g., images, text) into a latent distribution (mean and variance).  
   - Produces parameters of a probability distribution instead of a fixed vector.  
2. **Latent Space**  
   - Samples latent variables \(z\) from the learned distribution.  
   - Ensures continuity and smoothness for meaningful interpolation.  
3. **Decoder**  
   - Reconstructs data from sampled latent variables.  
   - Generates outputs resembling the original dataset.  

![ariational-Autoencoder-architecture](https://github.com/user-attachments/assets/79ce9f47-8900-423f-8c1a-2fc8f6c26102)

**Training Objective**  
- Uses **Evidence Lower Bound (ELBO)**:  
  - **Reconstruction loss**: Measures how well the decoder reconstructs input.  
  - **KL divergence**: Regularizes latent space by pushing learned distribution closer to a prior (usually Gaussian).  

**Applications**  
- Image synthesis (e.g., generating realistic faces).  
- Anomaly detection (identifying unusual patterns).  
- Representation learning (learning compressed, meaningful features).  
- Semi-supervised learning (leveraging unlabeled data).  

**Key Features**  
- Learns **continuous latent representations**.  
- Enables **controlled data generation** (e.g., tweaking latent variables to modify outputs).  
- Provides a probabilistic framework, unlike deterministic autoencoders.  
