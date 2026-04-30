from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


class Model:
    def __init__(self, n_estimators=5000, max_depth=4, learning_rate=0.01,
                 objective="binary:logistic", eval_metric="logloss",
                 tree_method="hist", random_state=42, early_stopping_rounds=50,
                 subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                 gamma=0.1, reg_alpha=0.1, reg_lambda=1.0):

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective=objective,
            eval_metric=eval_metric,
            tree_method=tree_method,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
        )

    def train(self, x_train, y_train, x_val, y_val):
        self.model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=False
        )

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        preds = self.model.predict(x)
        return accuracy_score(y, preds)

    def save(self, filename="titanic_model.json"):
        self.model.save_model(filename)

    def load(self, filename="titanic_model.json"):
        self.model.load_model(filename)