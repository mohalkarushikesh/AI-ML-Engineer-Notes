from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load the dataset
X, y = load_breast_cancer(return_X_y=True)

# split the data into trainning and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

clf = LogisticRegression(max_iter = 10000, random_state = 0)

clf.fit(X_train, y_train)


acc = accuracy_score(y_test, clf.predict(X_test)) * 100

print(f"Logistic Regression Accuracy Score: {acc: .2f}%")
