from Dataset import Dataset
from Model import Model

# --- Create Dataset and Model ---
dataset = Dataset()
model = Model()




# --- Train Model ---
model.train(dataset.x_train, dataset.y_train, dataset.x_val, dataset.y_val)
print("Validation accuracy:", model.score(dataset.x_val, dataset.y_val))
# --- Save Model --- 
model.model.save_model("titanic_model1.json")
