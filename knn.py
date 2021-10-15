#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

# Parsing the csv into a dataframe
df = pd.read_csv('merc.csv')

# Removing duplicates rows and rows with meaningless values
df["engineSize"] = df["engineSize"].replace(0, np.nan)
df["tax"] = df["tax"].replace(0, np.nan)
df["fuelType"] = df["fuelType"].replace("Other", np.nan)
df["transmission"] = df["transmission"].replace("Other", np.nan)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Modifying column dtypes to appropriate dtypes
df["model"] = df["model"].astype("category")
df["year"] = df["year"].astype("category")
df["price"] = df["price"].astype("int64")
df["transmission"] = df["transmission"].astype("category")
df["mileage"] = df["mileage"].astype("int64")
df["fuelType"] = df["fuelType"].astype("category")
df["tax"] = df["tax"].astype("int64")
df["mpg"] = df["mpg"].astype("float64")
df["engineSize"] = df["engineSize"].astype("float64")

# Removing outliers
df = df[((df["fuelType"] != "Hybrid") | (df["mpg"] > 10))]


#%%
# Removing uninteresting columns
df2 = df.drop(columns=["model", "year", "price", "transmission", "mileage", "tax", "engineSize"])
X = np.array(df2.drop(columns=["fuelType"]))
y = np.array(df2["fuelType"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

k = 10
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Measuring
print("Confussion matrix:")
print(f"{confusion_matrix(y_test, y_pred)}\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
print(f"{classification_report(y_test, y_pred)}\n")

# Plotting the scatter
plt.figure()
plt.scatter(X, y, marker="+")
plt.scatter(X_test, y_pred, marker="+")
plt.xlabel("Miles per gallon")
plt.ylabel("Fuel type")
plt.show()


#%%

def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def helper_predict(x, X_train, y_train):
    # print("breakpoint (helper_predict)")
    # Computing distances between x and others
    distances = [distance(x, x_train) for x_train in X_train]
    # Sorting with argsort
    k_indices = np.argsort(distances)[:k]
    # Finding labels of the k nearest neighbors and returning most common
    k_neighbor_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_neighbor_labels).most_common(1)
    return most_common[0][0]  # returns the label


def predict(X, X_train, y_train):
    y_pred = [helper_predict(x, X_train, y_train) for x in X]
    # print(y_pred)
    return np.array(y_pred)


df2 = df.drop(columns=["model", "year", "price", "transmission", "mileage", "tax", "engineSize"])
X = np.array(df2.drop(columns=["fuelType"]))
y = np.array(df2["fuelType"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

k = 1

y_pred = predict(X_test, X_train, y_train)

# Measuring
print("Confussion matrix:")
print(f"{confusion_matrix(y_test, y_pred)}\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
print(f"{classification_report(y_test, y_pred)}\n")


# Accuracy      = (correctly predicted observation) / (total observations)
#               = (TP+TN) / (TP+FP+FN+TN)
#
# Precision     = (correctly predicted positive observations) / (total predicted positive observations)
#               = TP / (TP+FP)
#
# Recall        = (correctly predicted positive observations) / (all observations in actual class)
#               = TP / (TP+FN)
#
# F1-Score = weighted average of Precision and Recall
#               = 2 * (Recall * Precision) / (Recall + Precision)
#
