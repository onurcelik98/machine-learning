#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from statistics import mode
from sklearn.ensemble import VotingClassifier

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

# Removing uninteresting columns
df2 = df.drop(columns=["price", "mileage", "tax"])
number_of_inputs = 5

# Splitting data
df_dummies = pd.get_dummies(df2[["fuelType", "year", "transmission"]])
df_dummies["mpg"] = df2["mpg"]
df_dummies["engineSize"] = df2["engineSize"]
X = np.array(df_dummies)
y = np.array(df2["model"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#%% SVM

# Training model and making predictions
model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Measuring
# print("Confussion matrix:")
# print(f"{confusion_matrix(y_test, y_pred)}\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
# print(f"{classification_report(y_test, y_pred)}\n")

# Plotting
if (number_of_inputs == 1):
    plt.figure()
    plt.scatter(X, y, marker="+")
    plt.scatter(X_test, y_pred, marker="+")
    plt.xlabel("Miles per gallon")
    plt.ylabel("Fuel type")
    plt.show()

else:
    plt.scatter(X[ : , 0], X[ : , 1], s =50, c="b")
    plt.show()


#%% Random forest

# Training model and making predictions
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Measuring
# print("Confussion matrix:")
# print(f"{confusion_matrix(y_test, y_pred)}\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
# print(f"{classification_report(y_test, y_pred)}\n")

# Plotting
if (number_of_inputs == 1):
    plt.figure()
    plt.scatter(X, y, marker="+")
    plt.scatter(X_test, y_pred, marker="+")
    plt.xlabel("Miles per gallon")
    plt.ylabel("Fuel type")
    plt.show()

else:
    plt.scatter(X[ : , 0], X[ : , 1], s =50, c="b")
    plt.show()


#%% AdaBoost

# Training model and making predictions
model = AdaBoostClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Measuring
# print("Confussion matrix:")
# print(f"{confusion_matrix(y_test, y_pred)}\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
# print(f"{classification_report(y_test, y_pred)}\n")

# Plotting
if (number_of_inputs == 1):
    plt.figure()
    plt.scatter(X, y, marker="+")
    plt.scatter(X_test, y_pred, marker="+")
    plt.xlabel("Miles per gallon")
    plt.ylabel("Fuel type")
    plt.show()

else:
    plt.scatter(X[ : , 0], X[ : , 1], s =50, c="b")
    plt.show()


#%% XGBoost

# Training model and making predictions
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Measuring
# print("Confussion matrix:")
# print(f"{confusion_matrix(y_test, y_pred)}\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
# print(f"{classification_report(y_test, y_pred)}\n")

# Plotting
if (number_of_inputs == 1):
    plt.figure()
    plt.scatter(X, y, marker="+")
    plt.scatter(X_test, y_pred, marker="+")
    plt.xlabel("Miles per gallon")
    plt.ylabel("Fuel type")
    plt.show()

else:
    plt.scatter(X[ : , 0], X[ : , 1], s =50, c="b")
    plt.show()


#%% Decision tree

# Training model and making predictions
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Measuring
# print("Confussion matrix:")
# print(f"{confusion_matrix(y_test, y_pred)}\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
# print(f"{classification_report(y_test, y_pred)}\n")

# Plotting
if (number_of_inputs == 1):
    plt.figure()
    plt.scatter(X, y, marker="+")
    plt.scatter(X_test, y_pred, marker="+")
    plt.xlabel("Miles per gallon")
    plt.ylabel("Fuel type")
    plt.show()

else:
    plt.scatter(X[ : , 0], X[ : , 1], s =50, c="b")
    plt.show()


#%% Logistic Regression

# Training model and making predictions
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Measuring
# print("Confussion matrix:")
# print(f"{confusion_matrix(y_test, y_pred)}\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
# print(f"{classification_report(y_test, y_pred)}\n")

# Plotting
if (number_of_inputs == 1):
    plt.figure()
    plt.scatter(X, y, marker="+")
    plt.scatter(X_test, y_pred, marker="+")
    plt.xlabel("Miles per gallon")
    plt.ylabel("Fuel type")
    plt.show()

else:
    plt.scatter(X[ : , 0], X[ : , 1], s =50, c="b")
    plt.show()


#%% k-NN

# Training model and making predictions
model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Measuring
# print("Confussion matrix:")
# print(f"{confusion_matrix(y_test, y_pred)}\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
# print(f"{classification_report(y_test, y_pred)}\n")

# Plotting
if (number_of_inputs == 1):
    plt.figure()
    plt.scatter(X, y, marker="+")
    plt.scatter(X_test, y_pred, marker="+")
    plt.xlabel("Miles per gallon")
    plt.ylabel("Fuel type")
    plt.show()

else:
    plt.scatter(X[ : , 0], X[ : , 1], s =50, c="b")
    plt.show()


#%% Ensemble (XGBoost, Random Forest, k-NN, Decision Tree)

model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train)

model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)

model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)

model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)


y_pred_xgb = model_xgb.predict(X_test)
y_pred_rf = model_rf.predict(X_test)
y_pred_knn = model_knn.predict(X_test)
y_pred_dt = model_dt.predict(X_test)


length = len(y_pred_xgb)
y_pred = [None] * length
for i in range(length):
    votes = [y_pred_rf[i], y_pred_dt[i], y_pred_knn[i]]
    y_pred[i] = mode(votes)

# Measuring
# print("Confussion matrix:")
# print(f"{confusion_matrix(y_test, y_pred)}\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
# print(f"{classification_report(y_test, y_pred)}\n")


#%% Ensemble (XGBoost, Random Forest, k-NN, Decision Tree)

# Hard Voting
vc_hard = VotingClassifier(
    estimators=[("xgb", XGBClassifier()),
                ("rf", RandomForestClassifier()),
                ("knn", KNeighborsClassifier()),
                ("dt", DecisionTreeClassifier())
    ],
    voting="hard")

vc_hard.fit(X_train, y_train)
y_pred = vc_hard.predict(X_test)

# Measuring
# print("Confussion matrix:")
# print(f"{confusion_matrix(y_test, y_pred)}\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
# print(f"{classification_report(y_test, y_pred)}\n")


#%% Ensemble (XGBoost, Random Forest, k-NN, Decision Tree)

# Soft Voting
vc_soft = VotingClassifier(
    estimators=[("xgb", XGBClassifier()),
                ("rf", RandomForestClassifier()),
                ("knn", KNeighborsClassifier()),
                ("dt", DecisionTreeClassifier())
    ],
    voting="soft")

vc_soft.fit(X_train, y_train)
y_pred = vc_soft.predict(X_test)

# Measuring
# print("Confussion matrix:")
# print(f"{confusion_matrix(y_test, y_pred)}\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
# print(f"{classification_report(y_test, y_pred)}\n")

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

