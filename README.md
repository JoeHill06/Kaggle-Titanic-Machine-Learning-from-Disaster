# Kaggle-Titanic-Machine-Learning-from-Disaster

### First Getting Our Data from kaggle cite
Creating the Dataset class and the object class to create the self.traning self.test and self.validation

### Understanding Our Data 
the values of correlation between each variable and survivability is:

Survived:Survived = 1.00
Survived:Fare = 0.27
Survived:Parch = 0.09
Survived:SibSp = -0.02
Survived:Age = -0.08
Survived:Age = -0.36

From This we can tell that the variable with the strongest correlation to survivability is the class that the passenger is apart off. The second most strongly correlated variabl is the Fare they paid to be onboard. The Third most correlated is Parch then Age then SibSP. 

### Model Design Ideas 





from xgboost import XGBClassifier
# read data
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], test_size=.2)
# create model instance
bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
# fit model
bst.fit(X_train, y_train)
# make predictions
preds = bst.predict(X_test)