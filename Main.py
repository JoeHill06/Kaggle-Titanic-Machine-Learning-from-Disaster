from Model import Model
from Dataset import Dataset
import random

# --- Load Model ---
loaded = Model()
loaded.load("titanic_model.json")

# --- Get Random Person ---
dataset = Dataset()
person = random.choice(dataset.validation)
person.show_training_data()

# --- Make Prediction ---
prediction = loaded.predict([person])
print(f" This passenger {"survived" if prediction[0] == 1 else "died"}")