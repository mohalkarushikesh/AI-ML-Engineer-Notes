
```mermaid
graph TD
    AIML[AI/ML Overview]

    %% Learning Paradigms
    AIML --> Paradigms[Learning Paradigms]
    Paradigms --> Supervised[Supervised Learning]
    Supervised --> Classification[Classification<br>Algorithms: Logistic Regression, SVM (Kernel-based), KNN, Naive Bayes<br>Evaluation: Accuracy, Precision, Recall, F1-Score, ROC-AUC]
    Supervised --> Regression[Regression<br>Algorithms: Linear, Polynomial, Ridge/Lasso (Regularization), SVR<br>Evaluation: MAE, MSE, RMSE, R-squared]
    Paradigms --> Unsupervised[Unsupervised Learning]
    Unsupervised --> Clustering[Clustering<br>Algorithms: K-Means, Hierarchical, DBSCAN, Gaussian Mixture Models<br>Evaluation: Silhouette Score, Davies-Bouldin Index]
    Unsupervised --> DimReduction[Dimensionality Reduction<br>Algorithms: PCA, t-SNE, Autoencoders<br>Applications: Feature Extraction for Images/Audio/Video]
    Paradigms --> Semi[Semi-Supervised Learning<br>Algorithms: Self-Training, Co-Training, Label Propagation<br>Uses: When Labeled Data is Scarce, e.g., Image Annotation]
    Paradigms --> RL[Reinforcement Learning<br>Algorithms: Q-Learning, SARSA, DDPG, PPO<br>Components: Agent, Environment, Rewards<br>Applications: Games, Robotics, Autonomous Driving (Video-based)]

    %% Deep Learning
    AIML --> DL[Deep Learning]
    DL --> NN[Neural Networks]
    NN --> ANN[Artificial Neural Networks<br>Basic Feedforward Networks<br>Frameworks: TensorFlow, PyTorch, Keras]
    NN --> CNN[Convolutional Neural Networks<br>For Image-based: Conv Layers, Pooling<br>Algorithms: AlexNet, ResNet, YOLO (Object Detection)<br>Applications: Computer Vision (CV), Image Classification/Segmentation]
    NN --> RNN[Recurrent Neural Networks<br>For Sequences: LSTM, GRU<br>Applications: Audio (Speech Recognition), Video (Action Recognition), Time-Series]
    RNN --> LLM[Large Language Models<br>Based on Transformers (Attention Mechanisms)<br>Algorithms: GPT, BERT, LLaMA<br>Applications: NLP, Text Generation, Chatbots<br>Frameworks: Hugging Face Transformers]
    DL --> CV[Computer Vision<br>Tasks: Object Detection, Facial Recognition, Semantic Segmentation<br>Algorithms: Faster R-CNN, U-Net<br>Image/Video-based Processing]

    %% Components
    AIML --> Components[Training Components]
    Components --> Optimizers[Optimizers<br>Algorithms: SGD, Adam, RMSprop, Adagrad<br>Uses: Gradient Descent in NN Training<br>Math: Calculus (Derivatives, Backpropagation)]
    Components --> Loss[Loss Functions<br>Examples: MSE (Regression), Cross-Entropy (Classification), Huber Loss<br>For RL: Value Loss, Policy Loss<br>Custom for Audio/Video: Perceptual Losses]
    Components --> Regularization[Regularization Techniques<br>L1/L2, Dropout, Early Stopping, Data Augmentation<br>Prevents Overfitting in DL Models<br>Applied in CNN/RNN for Images/Audio]

    %% Methods
    AIML --> Methods[Advanced Methods]
    Methods --> Ensemble[Ensemble Methods<br>Techniques: Bagging (Random Forest), Boosting (XGBoost, AdaBoost), Stacking<br>Tree-based: Decision Trees, Gradient Boosting Machines<br>Improves Accuracy in Classification/Regression]
    Methods --> Kernel[Kernel-based Methods<br>Algorithms: SVM with RBF/Polynomial Kernels<br>Uses: Non-linear Classification, e.g., Image Features]
    Methods --> Tree[Tree-based Methods<br>Algorithms: CART, ID3, C4.5, Extra Trees<br>Ensemble Integration: Random Forest for Feature Importance in CV]

    %% Foundations
    AIML --> Foundations[Foundations]
    Foundations --> Math[Math Foundations]
    Math --> Algebra[Linear Algebra<br>Vectors, Matrices, Eigenvalues<br>Uses: NN Weights, PCA in Dim Reduction]
    Math --> Calculus[Calculus<br>Derivatives, Integrals, Optimization<br>Essential for Backpropagation, Gradients in Optimizers]
    Foundations --> Stats[Statistics & Probability<br>Distributions (Normal, Poisson), Hypothesis Testing, Bayes' Theorem<br>Uses: Naive Bayes, Uncertainty in RL, Evaluation Metrics]

    %% Additional Suggestions
    AIML --> Apps[Applications & Extensions]
    Apps --> Frameworks[Frameworks & Libraries<br>ML: Scikit-learn (Supervised/Unsupervised)<br>DL: PyTorch, TensorFlow, MXNet<br>CV: OpenCV<br>NLP: spaCy, NLTK<br>Audio: Librosa, PyDub<br>Video: FFmpeg, MoviePy]
    Apps --> Media[Media Processing<br>Image: Augmentation (Flip, Rotate), GANs for Generation<br>Audio: Spectrograms, WaveNet<br>Video: Optical Flow, 3D CNNs<br>Multimodal: CLIP (Image+Text)]
    Apps --> OtherAlgo[Other Algorithms<br>Genetic Algorithms (Evolutionary), Fuzzy Logic<br>Hybrid: Neuro-Fuzzy Systems<br>Emerging: Federated Learning, Transfer Learning in DL]

    %% Connections for Cohesion
    DL --> Components
    RL --> Components
    CV --> CNN
    LLM --> RNN
    Ensemble --> Tree
    Kernel --> Supervised
    Stats --> Evaluation[Evaluation Techniques<br>Cross-Validation, Confusion Matrix, Bias-Variance Tradeoff<br>Applied Across All Paradigms]
    Foundations --> Evaluation
```
