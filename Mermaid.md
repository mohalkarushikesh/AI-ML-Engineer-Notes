```mermaid
graph TD
    A[Artificial Intelligence & Machine Learning]:::main

    %% Learning Paradigms
    A --> B[Learning Paradigms]
    B --> B1[Supervised Learning]
    B1 --> B1a[Classification<br/>• Logistic Regression<br/>• SVM (Linear/RBF Kernel)<br/>• Decision Trees<br/>• KNN • Naive Bayes<br/>• Neural Nets]
    B1 --> B1b[Regression<br/>• Linear / Polynomial<br/>• Ridge / Lasso / ElasticNet<br/>• SVR • Gradient Boosting Reg.]
    
    B --> B2[Unsupervised Learning]
    B2 --> B2a[Clustering<br/>• K-Means • Hierarchical<br/>• DBSCAN • Gaussian Mixtures]
    B2 --> B2b[Dimensionality Reduction<br/>• PCA • t-SNE • UMAP<br/>• Autoencoders • LDA]
    
    B --> B3[Semi-Supervised Learning<br/>• Self-Training<br/>• Label Propagation<br/>• Co-Training]
    B --> B4[Reinforcement Learning<br/>• Q-Learning • SARSA<br/>• DQN • PPO • A2C/A3C<br/>• Actor-Critic • Policy Gradient]

    %% Deep Learning Branch
    A --> C[Deep Learning]
    C --> C1[Neural Network Types]
    C1 --> ANN[ANN (MLP)]
    C1 --> CNN[CNN<br/>• LeNet • AlexNet<br/>• VGG • ResNet<br/>• EfficientNet • YOLO<br/>• U-Net (Segmentation)]
    C1 --> RNN[RNN Family<br/>• Vanilla RNN<br/>• LSTM • GRU<br/>• Seq2Seq • Attention]
    C1 --> Trans[Transformers<br/>• BERT • GPT • T5<br/>• ViT (Vision Transformer)<br/>• LLaMA • Gemini • Grok]
    C1 --> LLM[Large Language Models (LLM)]
    C1 --> GNN[Graph Neural Networks<br/>• GCN • GAT • GraphSAGE]
    
    C --> C2[Generative Models]
    C2 --> GAN[GANs<br/>• DCGAN • StyleGAN<br/>• CycleGAN • Pix2Pix]
    C2 --> VAE[VAE & Variants]
    C2 --> Diff[Diffusion Models<br/>• DDPM • Stable Diffusion<br/>• Sora (Video)]

    %% Computer Vision & Media
    A --> D[Domain-Specific]
    D --> CV[Computer Vision<br/>• Object Detection<br/>• Instance/Semantic Segmentation<br/>• Pose Estimation • OCR]
    D --> Audio[Audio / Speech<br/>• MFCC • Spectrograms<br/>• WaveNet • Whisper<br/>• HuBERT • AudioLM]
    D --> Video[Video Processing<br/>• Action Recognition<br/>• 3D CNN • Video Transformers<br/>• SlowFast • TimeSformer]
    D --> Multi[Multimodal<br/>• CLIP • DALL·E<br/>• Flamingo • ImageBind<br/>• LLaVA (Vision+Language)]

    %% Key Components
    A --> E[Core Components]
    E --> Opt[Optimizers<br/>• SGD • Momentum<br/>• Adam • AdamW<br/>• RMSprop • LAMB]
    E --> Loss[Loss Functions<br/>• MSE • Cross-Entropy<br/>• Dice • Contrastive<br/>• Perceptual • CTC]
    E --> Reg[Regularization<br/>• L1/L2 • Dropout<br/>• BatchNorm • LayerNorm<br/>• Early Stopping • Augmentation]
    
    %% Ensemble & Classical ML
    A --> F[Advanced Techniques]
    F --> Ens[Ensemble Methods<br/>• Bagging → Random Forest<br/>• Boosting → XGBoost, LightGBM, CatBoost<br/>• Stacking • Voting]
    F --> Kernel[Kernel Methods<br/>• SVM Kernels<br/>• Kernel PCA]
    F --> Tree[Tree-based<br/>• CART • ID3 • C4.5<br/>• Extra Trees]
    F --> Transfer[Transfer Learning<br/>• Fine-tuning • Feature Extraction<br/>• Domain Adaptation]
    F --> Few[Few-Shot / Zero-Shot Learning]

    %% Math & Evaluation
    A --> G[Foundations]
    G --> Math[Mathematics]
    Math --> Alg[Linear Algebra<br/>Matrices, Tensors, SVD]
    Math --> Calc[Calculus<br/>Gradients, Chain Rule, Backprop]
    Math --> Prob[Probability & Statistics<br/>Bayes, Distributions, Hypothesis Testing]
    
    G --> Eval[Evaluation Techniques<br/>• Cross-Validation<br/>• Confusion Matrix • ROC/PR AUC<br/>• BLEU • ROUGE • mAP<br/>• FID • IS (Generative)]

    %% Frameworks
    A --> H[Popular Frameworks]
    H --> Scikit[Scikit-learn]
    H --> TF[TensorFlow / Keras]
    H --> Torch[PyTorch]
    H --> HF[Hugging Face Transformers]
    H --> Others[Others<br/>JAX • MXNet • LightGBM<br/>OpenCV • Librosa • TorchAudio]

    %% Styling
    classDef main fill:#4a90e2,stroke:#333,color:white,font-weight:bold
    classDef category fill:#7ed321,stroke:#333,color:white
    class A,B,C,D,E,F,G,H category
