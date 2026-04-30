import csv
import random
import matplotlib.pyplot as plt
import numpy as np


class Dataset:
    def __init__(self, training_file="../titanic/train.csv", test_file="../titanic/test.csv"):
        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []
        self.x_test = []
        self.test_passenger_ids = []

        all_x = []
        all_y = []

        with open(training_file, "r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)
            raw_rows = list(reader)

        age_medians = self._compute_age_medians(raw_rows)

        for row in raw_rows:
            x_data = self.create_row(row, age_medians)
            y_data = int(row[1])
            all_x.append(x_data)
            all_y.append(y_data)

        # shuffle before split
        combined = list(zip(all_x, all_y))
        random.shuffle(combined)

        split_index = int(0.8 * len(combined))
        train_data = combined[:split_index]
        val_data = combined[split_index:]

        self.x_train = np.array([x for x, y in train_data], dtype=np.float32)
        self.y_train = np.array([y for x, y in train_data], dtype=np.int32)
        self.x_val = np.array([x for x, y in val_data], dtype=np.float32)
        self.y_val = np.array([y for x, y in val_data], dtype=np.int32)

        with open(test_file, "r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                self.test_passenger_ids.append(int(row[0]))
                self.x_test.append(self.create_test_row(row, age_medians))

        self.x_test = np.array(self.x_test, dtype=np.float32)

    def _compute_age_medians(self, rows):
        title_ages = {}
        for row in rows:
            name, age = row[3], row[5]
            if age == "":
                continue
            try:
                title = name.split(",")[1].strip().split(" ")[0].strip()
            except:
                title = "Unknown"
            title_ages.setdefault(title, []).append(float(age))
        return {title: float(np.median(ages)) for title, ages in title_ages.items()}



    def create_row(self, row, age_medians=None):
        passengerID, survived, Pclass, name, sex, age, SibSp, Parch, ticket, fare, cabin, embarked = row
        record = []

        # --- Simple Data ---
        record.append(int(Pclass) if Pclass != "" else 0)

        # --- For Name / Title ---
        titles = {
            "Master.": 1,
            "Mr.": 2,
            "Mrs.": 3,
            "Miss.": 4
        }

        try:
            name_part = name.split(",")[1].strip()
            extracted_title = name_part.split(" ")[0].strip()
            record.append(titles.get(extracted_title, 0))
        except:
            extracted_title = "Unknown"
            record.append(0)

        # --- For Sex ---
        gender = {
            "male": 1,
            "female": 2
        }
        record.append(gender.get(sex, 0))

        # --- For Age (impute missing with median age for same title) ---
        if age != "":
            age_val = float(age)
        elif age_medians:
            age_val = age_medians.get(extracted_title, age_medians.get("Mr.", 28.0))
        else:
            age_val = 28.0
        record.append(age_val)

        # --- IsChild ---
        record.append(1 if age_val < 12 else 0)

        # --- SibSp ---
        sibsp_val = int(SibSp) if SibSp != "" else 0
        record.append(sibsp_val)

        # --- Parch ---
        parch_val = int(Parch) if Parch != "" else 0
        record.append(parch_val)

        # --- FamilySize ---
        family_size = sibsp_val + parch_val + 1
        record.append(family_size)

        # --- IsAlone ---
        is_alone = 1 if family_size == 1 else 0
        record.append(is_alone)

        # --- Fare (log-transformed) ---
        fare_val = float(fare) if fare != "" else 0.0
        record.append(np.log1p(fare_val))

        # --- FarePerPerson ---
        record.append(np.log1p(fare_val / family_size))

        # --- Cabin / Deck ---
        deck_map = {
            "A": 1, "B": 2, "C": 3, "D": 4,
            "E": 5, "F": 6, "G": 7, "T": 8
        }
        if cabin != "":
            record.append(deck_map.get(cabin[0], 0))
        else:
            record.append(0)

        # --- Embarked ---
        embarked_map = {
            "S": 1,
            "C": 2,
            "Q": 3
        }
        record.append(embarked_map.get(embarked, 0))

        return record

    def create_test_row(self, row, age_medians=None):
        # Test CSV: PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
        passengerID, Pclass, name, sex, age, SibSp, Parch, ticket, fare, cabin, embarked = row
        dummy_survived = "0"
        return self.create_row([passengerID, dummy_survived, Pclass, name, sex, age, SibSp, Parch, ticket, fare, cabin, embarked], age_medians)

    def show_correlation_matrix(self):
        import torch

        feature_names = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]

        if not self.x_train:
            print("No training data available.")
            return

        data_tensor = torch.tensor(self.x_train, dtype=torch.float32)
        corr_matrix = torch.corrcoef(data_tensor.T)

        plt.figure(figsize=(8, 6))
        plt.imshow(corr_matrix.numpy(), interpolation="nearest")
        plt.colorbar()
        plt.xticks(range(len(feature_names)), feature_names, rotation=45)
        plt.yticks(range(len(feature_names)), feature_names)
        plt.title("Correlation Matrix")

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