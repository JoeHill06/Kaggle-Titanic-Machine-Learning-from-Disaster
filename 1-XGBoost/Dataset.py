import random
import matplotlib.pyplot as plt
from pprint import pformat
import csv
import torch

class Object():
    def __init__(self, passengerID, Pclass=None, Name=None, Sex=None, Age=None, SibSp=None, Parch=None, 
                Ticket=None, Fare=None, Cabin=None, Embarked=None, Survived=None):
        self.passengerID = passengerID
        self.p_class = Pclass
        self.name = Name
        self.sex = Sex
        self.age = Age
        self.sibsp = SibSp
        self.parch = Parch
        self.ticket = Ticket
        self.fare = Fare
        self.cabin = Cabin
        self.embarked = Embarked
        self.survived = Survived

    def show_training_data(self):
        data = {
            "Passenger ID": self.passengerID,
            "Pclass": self.p_class,
            "Name": self.name,
            "Sex": self.sex,
            "Age": self.age,
            "SibSp": self.sibsp,
            "Parch": self.parch,
            "Ticket": self.ticket,
            "Fare": self.fare,
            "Cabin": self.cabin,
            "Embarked": self.embarked,
            "Survived": self.survived
        }

        if self.survived == 1:
            colour = "\033[92m"
        elif self.survived == 0:
            colour = "\033[91m"
        else:
            colour = "\033[0m"

        reset = "\033[0m"

        print(f"{colour}\n--- Passenger Training Data ---")
        print(pformat(data, sort_dicts=False))
        print(reset, end="")


class Dataset(): # create training, validation and test data
    def __init__(self, training_file="titanic/train.csv", test_file="titanic/test.csv"):
        self.training = []
        self.test = []
        self.validation = []

        def clean(value):
            value = value.strip()
            return value if value != "" else None

        with open(training_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                passengerID = clean(row.get("PassengerId", ""))
                Pclass = clean(row.get("Pclass", ""))
                Name = clean(row.get("Name", ""))
                Sex = clean(row.get("Sex", ""))
                Age = clean(row.get("Age", ""))
                SibSp = clean(row.get("SibSp", ""))
                Parch = clean(row.get("Parch", ""))
                Ticket = clean(row.get("Ticket", ""))
                Fare = clean(row.get("Fare", ""))
                Cabin = clean(row.get("Cabin", ""))
                Embarked = clean(row.get("Embarked", ""))
                
                survived_raw = clean(row.get("Survived", ""))
                Survived = int(survived_raw) if survived_raw is not None else None

                self.training.append(
                    Object(
                        passengerID, Pclass, Name, Sex, Age, SibSp,
                        Parch, Ticket, Fare, Cabin, Embarked, Survived
                    )
                )

        random.shuffle(self.training)
        split_index = int(len(self.training) * 0.1)
        self.validation = self.training[:split_index]
        self.training = self.training[split_index:]

    def show_correlation_matrix(self): # shows a correlation matrix between all the variable
        feature_names = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Survived"]
        rows = []

        for passenger in self.training:
            values = [
                passenger.p_class,
                passenger.age,
                passenger.sibsp,
                passenger.parch,
                passenger.fare,
                passenger.survived
            ]

            # skip rows with missing numeric data
            if any(value is None for value in values):
                continue

            rows.append([float(value) for value in values])

        if not rows:
            print("No complete numeric rows available for correlation matrix.")
            return

        data_tensor = torch.tensor(rows, dtype=torch.float32)

        # transpose so each row is a feature
        corr_matrix = torch.corrcoef(data_tensor.T)

        plt.figure(figsize=(8, 6))
        plt.imshow(corr_matrix.numpy(), interpolation="nearest")
        plt.colorbar()
        plt.xticks(range(len(feature_names)), feature_names, rotation=45)
        plt.yticks(range(len(feature_names)), feature_names)
        plt.title("Correlation Matrix")

        # write the correlation numbers inside the squares
        for i in range(len(feature_names)):
            for j in range(len(feature_names)):
                plt.text(
                    j, i,
                    f"{corr_matrix[i, j]:.2f}",
                    ha="center",
                    va="center"
                )

        plt.tight_layout()
        plt.show()

