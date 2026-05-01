# Titanic — Machine Learning from Disaster

Kaggle competition entry using XGBoost with Optuna hyperparameter tuning.

**Final Kaggle score: 76.794%**

---

## Results

| Model | Validation Accuracy | Kaggle Score |
|-------|-------------------|--------------|
| Model 1 — Baseline XGBoost | 86% | — |
| Model 2 — Feature engineering + tuning | 87% | 76.3% |
| Model 3 — Further tuning | 84% | 70.0% |
| **Final submission** | — | **76.794%** |

---

## Project Structure

```
1-XGBoost/
├── Dataset.py       # Data loading and feature engineering
├── Model.py         # XGBClassifier wrapper
├── Train.py         # Train and save a model
├── Tune.py          # Optuna hyperparameter search (100 trials, 5-fold CV)
├── Main.py          # Generate Kaggle submission CSV
└── best_params.json # Best hyperparameters found by Optuna
titanic/
├── train.csv        # Kaggle training data (891 passengers)
├── test.csv         # Kaggle test data (418 passengers)
└── Titanic-Correlation-Matrix.png
```

---

## Feature Engineering

13 features extracted from the raw CSV columns:

| Feature | Description |
|---------|-------------|
| `Pclass` | Passenger class (1, 2, 3) |
| `Title` | Extracted from name — Master / Mr / Mrs / Miss / Other |
| `Sex` | Encoded male=1, female=2 |
| `Age` | Raw age; missing values imputed by **title-group median** |
| `IsChild` | 1 if age < 12 |
| `SibSp` | Siblings/spouses aboard |
| `Parch` | Parents/children aboard |
| `FamilySize` | SibSp + Parch + 1 |
| `IsAlone` | 1 if travelling solo |
| `Fare` | Log-transformed fare (`log1p`) |
| `FarePerPerson` | `log1p(Fare / FamilySize)` |
| `Deck` | Deck letter extracted from Cabin (A–G, T); 0 if missing |
| `Embarked` | Port encoded S=1, C=2, Q=3 |

Age imputation uses the median age for passengers sharing the same title, which outperforms a global median since age distributions differ strongly by title.

---

## Model

XGBClassifier with early stopping (50 rounds), trained on an 80/20 stratified split.

**Best hyperparameters (from Optuna):**

```json
{
  "max_depth": 6,
  "learning_rate": 0.0974,
  "subsample": 0.730,
  "colsample_bytree": 0.859,
  "min_child_weight": 5,
  "gamma": 0.664,
  "reg_alpha": 0.348,
  "reg_lambda": 1.789
}
```

---

## Workflow

### 1. Train a model
```bash
cd 1-XGBoost
python Train.py
```
Saves the trained model to `titanic_model1.json` and prints validation accuracy.

### 2. Tune hyperparameters
```bash
python Tune.py
```
Runs 100 Optuna trials with 5-fold stratified cross-validation. Saves the best parameters to `best_params.json` and retrains a final model on all available data.

### 3. Generate submission
```bash
python Main.py
```
Loads the saved model and writes `submission.csv` ready for Kaggle upload.

---

## Key Observations

- **Sex and Pclass** are the strongest predictors of survival — women and first-class passengers survived at much higher rates.
- **Title extraction** captures age/social-status information more precisely than raw age alone.
- **FamilySize / IsAlone** adds signal: solo travellers and very large families had lower survival rates.
- Log-transforming fare reduces the impact of outliers.
- Validation accuracy consistently overestimates Kaggle score due to overfitting to the small training set (891 rows), especially noticeable in Model 3 (84% val → 70% Kaggle).

---

## Dependencies

```
xgboost
scikit-learn
optuna
numpy
matplotlib
```

Install with:
```bash
pip install xgboost scikit-learn optuna numpy matplotlib
```
