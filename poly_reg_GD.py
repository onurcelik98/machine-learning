#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


def my_mean_squared_error(y_true, y_predicted):
    cost = np.sum((y_predicted - y_true) ** 2) / len(y_true)
    return cost


def my_r2_score(y, y_hat):
    # r2 = 1- SS_res / SS_tot
    SS_res = np.sum((np.array(y_hat)-np.array(y))**2)
    SS_tot = np.sum((np.array(y)-np.mean(np.array(y)))**2)
    return 1 - SS_res / SS_tot


def gradients(X, y, y_pred):
    m = X.shape[0]
    dw = (1 / m) * np.dot(X.T, (y_pred - y))
    db = (1 / m) * np.sum((y_pred - y))
    return dw, db


def x_transform(X, degrees):
    t = X.copy()
    for i in degrees:
        X = np.append(X, t ** i, axis=1)
    return X


def train(X, y, bs, degrees, max_iterations, alpha):
    # Performing preparations and initializations
    x = x_transform(X, degrees)
    m, n = x.shape
    w = np.zeros((n, 1))
    b = 0
    y = y.reshape(m, 1)
    losses = []

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
            y_hat = np.dot(xb, w) + b

            # Finding the gradients of the current prediction
            dw, db = gradients(xb, yb, y_hat)

            # Updating the parameters
            w -= alpha * dw
            b -= alpha * db

        # Calculating loss and appending it in the list.
        l = my_mean_squared_error(y, np.dot(x, w) + b)
        losses.append(l)

    # returning weights, bias and losses[].
    return w, b, losses


def predict(X, w, b, degrees):
    x1 = x_transform(X, degrees)
    return np.dot(x1, w) + b


# Generating synthetic data 1
noise = np.random.normal(0, 7, 1000).reshape(1000, 1) + 20 * np.random.rand(1000, 1)
X_1 = 5*(np.random.rand(1000, 1) - 0.5)
y_1 = -4*(X_1**3) + 4*(X_1**2) - 30 + noise


# Generating synthetic data 2
noise = np.random.normal(0, 7, 1000).reshape(1000, 1) + 20 * np.random.rand(1000, 1)
X_2 = 5*(np.random.rand(1000, 1) - 0.5)
y_2 = +4*(X_2**3) + 8*(X_2**2) - 30 + noise


# Generating synthetic data 3
noise = np.random.normal(0, 7, 1000).reshape(1000, 1) + 20 * np.random.rand(1000, 1)
X_3 = 5*(np.random.rand(1000, 1) - 0.5)
y_3 = -5*(X_3**3) - 12*(X_3**2) - 30 + noise


# Retrieving real data
df = pd.read_csv('merc.csv')
df2 = df.drop(columns=["model", "year", "transmission", "fuelType", "tax", "engineSize", "mpg"])
X_4 = df2["mileage"].ravel()

y_4 = df2["price"].ravel()

temp1, X_4, temp2, y_4 = train_test_split(X_4, y_4, test_size=0.2)
X_4.shape = (len(X_4), 1)
y_4.shape = (len(y_4), 1)

X, y = X_4, y_4


#%%
# Training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
w, b, l = train(X_train, y_train, bs=100, degrees=[2, 3], max_iterations=1000, alpha=0.01)


# Making predictions
y_pred = predict(X_test, w, b, [2, 3])
X_y = pd.DataFrame({"X_test": pd.Series(X_test.ravel()), "y_pred": pd.Series(y_pred.ravel())})\
    .sort_values(by="X_test", ascending=True)


# Plotting the polynomial along with scatter plot
fig1 = plt.figure(figsize=(10.24, 7.68))
plt.plot(X, y, 'y.')
plt.plot(X_y["X_test"], X_y["y_pred"])
plt.legend(["Data", "Polynomial predictions"])
plt.xlabel("X")
plt.ylabel("y")
plt.title("Polynomial regression")
plt.show()


# Measuring accuracy
my_rmse = np.sqrt(my_mean_squared_error(y_test, y_pred))
my_r2 = r2_score(y_test, y_pred)
print(f"RMSE     = {my_rmse}")
print(f"R2 score = {my_r2}")


#%%
# Preparing the training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
poly_features = PolynomialFeatures(degree=3)
X_train_poly = poly_features.fit_transform(X_train)

y_train, y_test = y_train.ravel(), y_test.ravel()


# Fitting the transformed features to Linear Regression
poly_model = SGDRegressor(max_iter=100000, eta0=0.0000001, tol=None, learning_rate="adaptive")
poly_model.fit(X_train_poly, y_train)


# Making predictions
y_pred = poly_model.predict(poly_features.fit_transform(X_test))
X_y = pd.DataFrame({"X_test": pd.Series(X_test.ravel()), "y_pred": pd.Series(y_pred.ravel())})\
    .sort_values(by="X_test", ascending=True)


# Plotting
fig2 = plt.figure(figsize=(10.24, 7.68))
plt.plot(X, y, 'c.')
plt.plot(X_y["X_test"], X_y["y_pred"], color="red")
plt.legend(["Data", "Polynomial predictions"])
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.title("Polynomial regression")
plt.show()


# Measuring accuracy
my_rmse = np.sqrt(my_mean_squared_error(y_test, y_pred))
my_r2_test = my_r2_score(y_test, y_pred)
print(f"RMSE     = {my_rmse}")
print(f"R2 score = {my_r2_test}")
