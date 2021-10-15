import pandas as pd
import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils.fixes import parse_version

# Initial setup (unused)
# print(pd.__version__)
plt.close("all")
rg = np.random.default_rng(1)
np.set_printoptions(threshold=sys.maxsize)

# Parsing the csv into a dataframe
df = pd.read_csv('merc.csv')

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

# Removing duplicates rows and rows with meaningless values, remapping categoricals
df["engineSize"] = df["engineSize"].replace(0, np.nan)
df["tax"] = df["tax"].replace(0, np.nan)
df["fuelType"] = df["fuelType"].replace("Other", np.nan)
df["transmission"] = df["transmission"].replace("Other", np.nan)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df["transmission"] = df["transmission"].map({'Manual': 0, 'Semi-Auto': 1, 'Automatic': 2})
df["fuelType"] = df["fuelType"].map({'Diesel': 0, 'Petrol': 1, 'Hybrid': 2})

# Removing outliers
# df = df[(df["mpg"] > 5.0) & (df["mpg"] < 100.0)]

# print(df)
# f = open("temp_data.txt", "w")
# f.write(df.to_string())
# f.close()
# print(df.sort_values(by="mpg", ascending=True))

# print(df.sort_values(by="mpg", ascending=True))
# plt.scatter(x=df["engineSize"], y=df["mpg"])
# plt.xlabel("Engine size")
# plt.ylabel("Miles per gallon")
# plt.savefig('plots/scatter_engineSize_vs_mpg')
# plt.show()


# Helper for plot labeling
def label(x):
    return {
        0: ("Engine size", "Miles per gallon"),
        1: ("Mileage", "Price"),
        -1: ("Mileage", "Price"),
        2: ("Engine size", "Tax"),
        3: ("Year", "Mileage"),
        -3: ("Year", "Mileage"),
    }[x]


# Helper for plot saving
def saveName(x):
    return {
        0:"plots/my_fit_engineSize_vs_mpg",
        1: "plots/my_fit_mileage_vs_price",
        -1: "plots/my_fit_mileage_vs_price",
        2: "plots/my_fit_engineSize_vs_tax",
        3: "plots/my_fit_year_vs_mileage",
        -3: "plots/my_fit_year_vs_mileage"
    }[x]


#%% Test
# df = df.sort_values(by="engineSize", ascending=True)
# print(df)
# f = open("temp_data.txt", "w")
# f.write(df.to_string())
# f.close()
#
#
# df.hist(figsize=(19.20, 10.80), bins=100)
# plt.show()
#
#
# print(df[df["tax"] >= 350].sort_values(by="tax", ascending=True).size / df.size)


#%% Stochastic Gradient Descent Algorithm

def my_mean_squared_error(y_true, y_predicted):
    cost = np.sum((y_predicted - y_true) ** 2) / len(y_true)
    return cost


def gradient_descent(x, y, iterations=10000, learning_rate=0.000000001, stopping_threshold=1e-6):
    current_weight = 0.1
    current_bias = 0.01
    iterations = iterations
    learning_rate = learning_rate
    n = float(len(x))

    costs = []
    weights = []
    previous_cost = None

    for i in range(iterations):
        print(f"Iteration {i}:")

        # Making predictions
        y_predicted = (current_weight * x) + current_bias
        # print(f"y_predicted = {y_predicted}")

        # Calculating the current cost
        current_cost = my_mean_squared_error(y, y_predicted) / 2.0
        print(f"current_cost = {current_cost}")

        # Checking if cost magnitude is still significant
        # if previous_cost and abs(previous_cost - current_cost) <= stopping_threshold:
        #     break

        previous_cost = current_cost
        costs.append(current_cost)
        weights.append(current_weight)

        # Calculating the gradients
        weight_derivative = (1.0 / n) * sum(x * (y_predicted - y))
        bias_derivative = (1.0 / n) * sum(y_predicted - y)
        print(f"weight_derivative = {weight_derivative}")
        print(f"bias_derivative = {bias_derivative}")

        # Updating weights and bias
        current_weight = current_weight - (learning_rate * weight_derivative)
        current_bias = current_bias - (learning_rate * bias_derivative)

        print()

    return current_weight, current_bias


#%% Choosing X and y (engine size vs mpg)

mode = 0
# Dropping unneeded columns
df2 = df.drop(columns=["model", "year", "transmission", "fuelType", "tax", "price", "mileage"])

X = np.array(df2.drop(columns=["mpg"]))
y = np.array(df2["mpg"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#%% Choosing X and y (mileage vs price)

mode = 1
# Dropping unneeded columns
df2 = df.drop(columns=["model", "year", "transmission", "fuelType", "tax", "mpg", "engineSize"])
df["price"] = df["price"] / 1000.0
df["mileage"] = df["mileage"] / 1000.0

X = np.array(df2.drop(columns=["price"]), dtype="object")
y = np.array(df2["price"], dtype="object")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#%% Choosing X and y (engine size vs tax)

mode = 2
# Dropping unneeded columns
df2 = df.drop(columns=["model", "year", "transmission", "fuelType", "price", "mpg", "mileage"])

X = np.array(df2.drop(columns=["tax"]))
y = np.array(df2["tax"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#%% Choosing X and y (year vs mileage)

mode = 3
# Dropping unneeded columns
df2 = df.drop(columns=["model", "tax", "transmission", "fuelType", "price", "mpg", "engineSize"])
df2 = df2[df2["mileage"] < 200000]
df2["mileage"] = df2["mileage"] / 1000.0
# df2["year"] = df2["year"] / 1000.0

X = np.array(df2.drop(columns=["mileage"]), dtype="object")
y = np.array(df2["mileage"], dtype="object")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



#%% Model training

# Estimating weight and bias using gradient descent
estimated_weight, estimated_bias = gradient_descent(X_train.ravel(), y_train)
print(f"Estimated Weight: {estimated_weight}\nEstimated Bias: {estimated_bias}")


#%% Predictions

# Making predictions using estimated parameters
y_pred = estimated_weight * X_test + estimated_bias

# Scaling back if needed
if (mode in [1, 3]):
    # X *= 1000
    y *= 1000
    mode *= -1

# Plotting the regression line
x_point = np.linspace(min(X)-0.1,max(X)+0.1)
y_point = estimated_weight * x_point + estimated_bias

# Plotting the scatter chart
plt.figure(figsize=(8, 6))
plt.scatter(X, y, marker='o', color='red')
plt.plot(x_point, y_point,
         color='blue', markerfacecolor='red', markersize=10, linestyle='dashed')
axis_labels = label(mode)
plt.xlabel(axis_labels[0])
plt.ylabel(axis_labels[1])
save_name = saveName(mode)
plt.savefig(save_name)
plt.show()

# Calculating rmse with my function and sklearn's
my_rmse = np.sqrt(my_mean_squared_error(y_test, y_pred.ravel()))
print(my_rmse)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)


#%% Stochastic Gradient Descent Algorithm

# Initializing the model and the input-output sets for training and testing
model = SGDRegressor(max_iter=10000, eta0=0.000000001, learning_rate="adaptive", fit_intercept=True)  # learning_rate="constant" commented out, default="invscaling"

df2 = df.drop(columns=["model", "year", "transmission", "fuelType", "tax", "mpg", "engineSize"])
df2 = df2[df2["mileage"] < 200000]
df2 = df2[df2["mileage"] > 1]
# df2 = df2[df2["price"] < 60000]
# df2["mileage"] = 1.0 / df2["mileage"]

X = np.array(df2.drop(columns=["price"]))
y = np.array(df2["price"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# print(X)
# print(y)

# plt.figure()
# plt.scatter(X, y)
# plt.xlabel("Mileage")
# plt.ylabel("Price")
# plt.show()

# Training the model
model.fit(X_train, y_train)

# Making predictions on test input set
predictions = model.predict(X_test)
y_predicted = pd.DataFrame(predictions)
# print(y_predicted)
my_rmse = np.sqrt(my_mean_squared_error(y_test, predictions))
print(my_rmse)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(rmse)

plt.scatter(X, y, color='g')
plt.plot(X, model.predict(X), color='r')
plt.xlabel("Mileage")
plt.ylabel("Price")
# plt.savefig("plots/final")
plt.show()
