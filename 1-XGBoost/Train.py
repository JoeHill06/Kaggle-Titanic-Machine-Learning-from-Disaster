from Dataset import Dataset
from Model import Model

# --- Create Dataset and Model ---
dataset = Dataset()
model = Model()


# --- Train Model ---
model.train(dataset.training)
print("Validation accuracy:", model.score(dataset.validation))

# --- Save Model --- 
model.model.save_model("titanic_model.json")
