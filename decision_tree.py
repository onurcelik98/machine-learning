#%%
import numpy as np
import pandas as pd
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

# Removing uninteresting columns
df2 = df.drop(columns=["price", "mileage", "tax"])
number_of_inputs = 5

# Splitting data
df_dummies = pd.get_dummies(df2[["model", "year", "transmission"]])
df_dummies["mpg"] = df2["mpg"]
df_dummies["engineSize"] = df2["engineSize"]
X = np.array(df_dummies)
y = np.array(df2["fuelType"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#%% Decision tree implementation

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        # internal (decision) node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        # leaf node
        self.value = value


class MyDecisionTreeClassifier:
    def __init__(self, min_samples_split=3, max_depth=3):
        self.root = None
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for c in class_labels:
            p_i = len(y[y == c]) / len(y)  # probability of picking c
            entropy += -p_i * np.log2(p_i)  # total entropy
        return entropy

    def information_gain(self, parent, l_child, r_child):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain

    def get_best_split(self, dataset, num_features):
        best_split = {}  # dictionary
        max_info_gain = -np.inf  # minimum

        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
                dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
                # check if children are not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y)
                    # update the best split if needed
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
        return best_split

    def build_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)

        # split until stopping conditions are met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_features)
            # check if information gain is positive
            if best_split["info_gain"] > 0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["info_gain"])

        # no more splits, decide on the label
        list_Y = list(Y)
        leaf_value = max(list_Y, key=list_Y.count)
        # return leaf node
        return Node(value=leaf_value)

    def print_tree(self, tree=None, indent="\t"):  # debugging purposes
        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print("X_" + str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + "\t")
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + "\t")

    def fit(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    def find(self, x, tree):  # predict recursive helper
        if tree.value is not None:  # leaf node
            return tree.value
        feature_val = x[tree.feature_index]  # internal node
        if feature_val <= tree.threshold:
            return self.find(x, tree.left)
        else:
            return self.find(x, tree.right)

    def predict(self, X):
        predictions = [self.find(x, self.root) for x in X]
        return predictions


#%% Training the model

y_train.shape = (y_train.shape[0], 1)

model = MyDecisionTreeClassifier(max_depth=7)
model.fit(X_train, y_train)
# classifier.print_tree()


#%% Making predictions and measuring

y_pred = model.predict(X_test)

# Measuring
print("Confussion matrix:")
print(f"{confusion_matrix(y_test, y_pred)}\n")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
print(f"{classification_report(y_test, y_pred)}\n")
