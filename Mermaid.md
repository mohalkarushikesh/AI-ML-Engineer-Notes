graph TD
    A[Artificial Intelligence & Machine Learning]:::main

    A --> B[Learning Paradigms]
    B --> B1[Supervised Learning]
    B1 --> B1a["Classification<br/>• Logistic Regression<br/>• SVM (Linear/RBF Kernel)<br/>• Decision Trees<br/>• KNN • Naive Bayes<br/>• Neural Nets"]
    B1 --> B1b["Regression<br/>• Linear / Polynomial<br/>• Ridge / Lasso / ElasticNet<br/>• SVR • Gradient Boosting"]

    B --> B2[Unsupervised Learning]
    B2 --> B2a["Clustering<br/>• K-Means • Hierarchical<br/>• DBSCAN • Gaussian Mixtures"]
    B2 --> B2b["Dim. Reduction<br/>• PCA • t-SNE • UMAP<br/>• Autoencoders • LDA"]

    B --> B3["Semi-Supervised<br/>• Self-Training<br/>• Label Propagation"]
    B --> B4["Reinforcement Learning<br/>• Q-Learning • DQN • PPO<br/>• Actor-Critic • Policy Gradient"]

    A --> C[Deep Learning]
    C --> C1[Neural Network Types]
    C1 --> ANN["ANN (MLP)"]
    C1 --> CNN["CNN<br/>• ResNet • YOLO • U-Net"]
    C1 --> RNN["RNN / LSTM / GRU"]
    C1 --> Trans["Transformers<br/>• BERT • GPT • LLaMA<br/>• ViT • Grok"]
    C1 --> LLM[Large Language Models]
    C1 --> GNN["Graph Neural Nets<br/>• GCN • GAT"]

    C --> Gen[Generative Models]
    Gen --> GAN["GANs<br/>• StyleGAN • Stable Diffusion"]
    Gen --> Diff[Diffusion Models]

    A --> D[Domains]
    D --> CV[Computer Vision]
    D --> Audio[Audio / Speech]
    D --> Video[Video Processing]
    D --> Multi[Multimodal<br/>• CLIP • DALL·E • LLaVA]

    A --> E[Core Components]
    E --> Opt["Optimizers<br/>• SGD • Adam • AdamW"]
    E --> Loss["Loss Functions<br/>• MSE • Cross-Entropy<br/>• Contrastive • CTC"]
    E --> Reg["Regularization<br/>• L1/L2 • Dropout<br/>• BatchNorm • Augmentation"]

    A --> F[Advanced Techniques]
    F --> Ens["Ensemble<br/>• Random Forest<br/>• XGBoost • LightGBM"]
    F --> Transfer[Transfer / Few-Shot Learning]

    A --> G[Foundations]
    G --> Math["Math<br/>• Linear Algebra<br/>• Calculus<br/>• Probability"]
    G --> Eval["Evaluation<br/>• Accuracy • F1 • AUC<br/>• BLEU • FID"]

    A --> H["Frameworks<br/>• Scikit-learn<br/>• PyTorch<br/>• TensorFlow<br/>• Hugging Face"]

    classDef main fill:#4a90e2,stroke:#333,color:white,font-weight:bold
    classDef category fill:#7ed321,stroke:#333,color:white
    class A,B,C,D,E,F,G,H category
