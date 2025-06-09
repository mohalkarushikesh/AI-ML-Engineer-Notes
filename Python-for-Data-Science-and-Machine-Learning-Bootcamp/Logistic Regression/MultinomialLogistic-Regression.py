# Multinomial Logistic Regression

# Imports
from sklearn.model_selection import train_test_split
from sklearn import linear_model, metrics, datasets

# load dataset
digits = datasets.load_digits()

X = digits.data
y = digits.target

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=1)

# fit the model
reg = linear_model.LogisticRegression(max_iter = 10000, random_state=1)
reg.fit(X_train, y_train)

# prediction
y_pred = reg.predict(X_test)

# evaluation
pred = metrics.accuracy_score(y_test, y_pred) * 100

print(f"Logistic Regression model Accuracy: {pred : .2f}%")
