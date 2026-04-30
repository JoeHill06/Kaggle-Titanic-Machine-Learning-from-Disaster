from Dataset import Dataset
from Model import Model
import random

# --- Create Dataset, Model
dataset = Dataset()
loaded = Model()
loaded.load("titanic_model.json")

# --- Get Random Person ---
#index = random.randrange(len(dataset.x_val))
#person_data = [dataset.x_val[index]]
#actual = dataset.y_val[index]

# --- Predict Person
#prediction = loaded.predict(person_data)[0]
#print("Features:", person_data[0])
#print("Actual:", actual)
#print("Prediction:", prediction)

# --- Predict Submission ---
predictions = loaded.predict(dataset.x_test)

file = "submission.csv"   # saves in current folder
with open(file, "w") as f:
    f.write("PassengerId,Survived\n")
    
    for i in range(len(predictions)):
        passengerID = dataset.test_passenger_ids[i]
        prediction = predictions[i]
        f.write(f"{passengerID},{prediction}\n")