**Decision Trees and Random Forests **

Decision Trees: is supervised learning algorithm used for regression and classification problems
- In decision tree, flow-chart like structure is build where each internal node denotes the features. rules are denoted using the branches and the leaves denotes the final result of the algorithm.
- use when interpretability is important and you need  a simple and easy to understand model 
- If computational efficiency is a concern and you have a small dataset DT might be more appropriate 

Randome Forest: is supervised learning algorithm used for regression and classification tasks
- uses esemble learning (combining multiple models/classifiers to solve the complex problem and to improve overall accuracy score of the model)
- Multiple decision trees are built by considering diff subset of given data the avg of all those to increase accuracy of the model
- As the no of decision trees increase accuracy increases and overfitting also reduces 
- use when you want better generalization performance, robustness to overfitting and improve accuracy especially on the complex datasets with high dimentional feature spaces
- If you have large dataset with complex relationship between the features and labels

  Property

Random Forest

Decision Tree

Nature

Ensemble of multiple decision trees

Single Decision Tree

Interpretability

Less interpretable due to ensemble nature.

Highly interpretable.

Overfitting

Due to ensemble averaging it is less prone to overfitting.

More prone to overfitting specially in case of deep trees.

Training Time

Since multiple trees are constructed, training time becomes more, and training speed becomes less.

A single tree needs to be built and trained, hence faster in comparison.

Stability to change

Since overall average is taken due to ensemble, it is more stable to change.

It becomes quite sensitive to variation in data.

Predictive Time

Multiple predictions, hence longer prediction time and slower prediction speed.

Faster prediction as compared to random forest, since a single prediction is made.

Performance             Generally performs well on large datasets.      It can perform well on small and large dataset as well.

Handling Outliers       Due to ensemble averaging more robust to outliers.      It is more susceptible to outliers.

Feature Importance      Do not provide feature score directly rather uses ensemble to decide feature score.      Provide feature score directly which are less reliable.
