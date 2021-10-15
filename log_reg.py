#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

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

# df2["fuelType"] = pd.Categorical(df2["fuelType"], categories=["Petrol", "Diesel", "Hybrid"], ordered=True)


#%%

# Removing uninteresting columns
df2 = df.drop(columns=["model", "year", "price", "transmission", "mileage", "tax", "engineSize"])

# Splitting data
X = np.array(df2.drop(columns=["fuelType"]))
y = np.array(df2["fuelType"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Training model and making predictions
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Plotting the scatter
plt.figure()
plt.scatter(X, y, marker="+")
plt.scatter(X_test, y_pred, marker="+")
plt.xlabel("Miles per gallon")
plt.ylabel("Fuel type")
plt.show()

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

#%%

def gradients(X, y, y_pred):
    m = X.shape[0]
    dw = (1 / m) * np.dot(X.T, (y_pred - y))
    db = (1 / m) * np.sum((y_pred - y))
    return dw, db


def train(X, y, bs, max_iterations, alpha):
    # Performing preparations and initializations
    x = X
    m, n = x.shape
    w = np.zeros((n, 1))
    b = 0
    y = y.reshape(m, 1)
    # print(x)
    # print(y)

    # Main loop
    for epoch in range(max_iterations):
        # Using mini-batch
        for i in range((m - 1) // bs + 1):
            # Declaring current batch
            start_i = i * bs
            end_i = start_i + bs
            xb = x[start_i:end_i]
            yb = y[start_i:end_i]

            # Calculating the current prediction
            linear_model = np.dot(xb, w) + b
            y_hat = 1.0 / (1 + np.exp(-linear_model))  # Sigmoid

            # Finding the gradients of the current prediction
            dw, db = gradients(xb, yb, y_hat)

            # Updating the parameters
            w -= alpha * dw
            b -= alpha * db

    # returning weights, bias and losses[].
    return w, b


def predict(X, w, b):
    x1 = X
    linear_model = np.dot(x1, w) + b
    y_predicted = 1 / (1 + np.exp(-linear_model))
    y_predicted_cls = []
    for e in y_predicted:
        if e <= 0.33:
            y_predicted_cls.append(0)
        elif e > 0.33 and e < 0.67:
            y_predicted_cls.append(1)
        else:
            y_predicted_cls.append(2)
    return np.array(y_predicted_cls)


# Removing uninteresting columns
df3 = df.drop(columns=["model", "year", "price", "transmission", "mileage", "tax", "engineSize"])
# df3 = df3[df3["fuelType"] != "Diesel"]
df4 = df.drop(columns=["price", "mileage", "tax"])
df_dummies = pd.get_dummies(df4[["model", "year", "transmission"]])
df_dummies["mpg"] = df4["mpg"]
df_dummies["engineSize"] = df4["engineSize"]
X = np.array(df_dummies)
y = np.array(df4["fuelType"])
le_fuelType = LabelEncoder()

X = np.array(df_dummies)
y = np.array(le_fuelType.fit_transform(df4["fuelType"]))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
w, b = train(X_train, y_train, bs=100, max_iterations=1000, alpha=0.01)


# Making predictions
y_pred = predict(X_test, w, b)
X_y = pd.DataFrame({"X_test": pd.Series(X_test.ravel()), "y_pred": pd.Series(y_pred.ravel())})\
    .sort_values(by="X_test", ascending=True)


# Measuring
print("Confussion matrix:")
print(f"{confusion_matrix(y_test, y_pred)}\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
print(f"{classification_report(y_test, y_pred)}\n")

# print(y_test)
# print(y_pred)


# plt.figure()
# plt.scatter(X, y, marker="+")
# plt.scatter(X_test, y_pred, marker="+")
# plt.xlabel("Miles per gallon")
# plt.ylabel("Fuel type")
# plt.show()


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
