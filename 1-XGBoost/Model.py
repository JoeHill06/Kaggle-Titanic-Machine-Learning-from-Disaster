from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


class Model:
    def __init__(self, n_estimators=200,max_depth=4,learning_rate=0.05,
                 objective="binary:logistic",eval_metric="logloss",
                 tree_method="hist",random_state=42):
        
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42
        )

    def object_to_features(self, obj):
        sex = 1 if obj.sex == "male" else 0 if obj.sex == "female" else -1
        embarked_map = {"S": 0, "C": 1, "Q": 2}
        embarked = embarked_map.get(obj.embarked, -1)

        return [
            float(obj.p_class) if obj.p_class is not None else 0.0,
            float(obj.age) if obj.age is not None else 0.0,
            float(obj.sibsp) if obj.sibsp is not None else 0.0,
            float(obj.parch) if obj.parch is not None else 0.0,
            float(obj.fare) if obj.fare is not None else 0.0,
            float(sex),
            float(embarked),
        ]

    def build_training_data(self, passengers):
        X = []
        y = []

        for p in passengers:
            if p.survived is None:
                continue

            X.append(self.object_to_features(p))
            y.append(int(p.survived))

        return X, y

    def train(self, passengers):
        X, y = self.build_training_data(passengers)
        self.model.fit(X, y)

    def predict(self, passengers):
        X = [self.object_to_features(p) for p in passengers]
        return self.model.predict(X)

    def score(self, passengers):
        X, y = self.build_training_data(passengers)
        preds = self.model.predict(X)
        return accuracy_score(y, preds)
    
    def load(self, filename="titanic_model.json"):
        self.model.load_model(filename)