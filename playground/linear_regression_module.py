
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import datasets

def perform_linear_regression_on_iris():
    # Load the Iris dataset
    iris = datasets.load_iris()
    data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

    # Assuming we want to predict the sepal length ('sepal length (cm)') based on other features
    target_feature = 'sepal length (cm)'
    X = data.drop([target_feature, 'target'], axis=1)
    y = data[target_feature]

    # Split the dataset into training and testing sets
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    return model
