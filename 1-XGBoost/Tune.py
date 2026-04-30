import json
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from Dataset import Dataset

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, x, y):
    params = {
        "n_estimators": 2000,
        "max_depth":          trial.suggest_int("max_depth", 2, 7),
        "learning_rate":      trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":   trial.suggest_int("min_child_weight", 1, 10),
        "gamma":              trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha":          trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda":         trial.suggest_float("reg_lambda", 0.5, 2.0),
        "objective":          "binary:logistic",
        "eval_metric":        "logloss",
        "tree_method":        "hist",
        "early_stopping_rounds": 30,
        "random_state":       42,
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(x, y):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = XGBClassifier(**params)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        scores.append(accuracy_score(y_val, model.predict(x_val)))

    return np.mean(scores)


if __name__ == "__main__":
    dataset = Dataset()
    x = np.concatenate([dataset.x_train, dataset.x_val])
    y = np.concatenate([dataset.y_train, dataset.y_val])

    n_trials = 100
    study = optuna.create_study(direction="maximize")

    print(f"Running {n_trials} trials...\n")

    for i in range(n_trials):
        study.optimize(lambda trial: objective(trial, x, y), n_trials=1)
        best = study.best_value
        current = study.trials[-1].value
        print(f"Trial {i+1:>3}/{n_trials}  accuracy: {current:.4f}  best: {best:.4f}")

    print("\n--- Best result ---")
    print(f"Accuracy: {study.best_value:.4f}")
    print("Params:")
    for key, val in study.best_params.items():
        print(f"  {key}: {val}")

    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print("\nSaved params to best_params.json")

    print("Training final model on all data...")
    final_model = XGBClassifier(
        **study.best_params,
        n_estimators=2000,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        early_stopping_rounds=30,
        random_state=42,
    )
    split = int(0.9 * len(x))
    final_model.fit(x[:split], y[:split], eval_set=[(x[split:], y[split:])], verbose=False)
    final_model.save_model("titanic_model.json")
    print("Saved model to titanic_model.json")
