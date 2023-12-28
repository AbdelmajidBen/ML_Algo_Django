from django.shortcuts import render
from .linear_regression_module import perform_linear_regression_on_iris
from .knn import perform_knn_on_iris
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression


def linear_regression_result1(request):
    # Call your linear regression function
    trained_model = perform_linear_regression_on_iris()

    # Load the Iris dataset
    iris = datasets.load_iris()
    data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

    # Assuming we want to predict the sepal length ('sepal length (cm)') based on other features
    target_feature = 'sepal length (cm)'
    X = data.drop([target_feature, 'target'], axis=1)
    y = data[target_feature]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Make predictions on the test set
    y_pred = trained_model.predict(X_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)

    # Pass relevant data to the template
    coefficients = trained_model.coef_
    intercept = trained_model.intercept_

    context = {
        'coefficients': coefficients,
        'intercept': intercept,
        'mse': mse,
    }

    # Render the template with the provided context
    return render(request, 'index.html', context)



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import datasets
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend to avoid threading issues
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from django.shortcuts import render
from django.http import HttpResponse
from io import BytesIO, StringIO
import base64

# Assuming this function performs linear regression training
def perform_linear_regression_on_iris():
    iris = datasets.load_iris()
    data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

    # Assuming we want to predict the sepal length ('sepal length (cm)') based on other features
    target_feature = 'sepal length (cm)'
    X = data.drop([target_feature, 'target'], axis=1)
    y = data[target_feature]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    return model, X_test, y_test

def linear_regression_result(request):
    # Call your linear regression function
    trained_model, X_test, y_test = perform_linear_regression_on_iris()

    # Check the sizes of X_test and y_test
    if len(X_test) != len(y_test):
        error_message = "Error: X_test and y_test must have the same size."
        return render(request, 'error.html', {'error_message': error_message})

    # Make predictions on the test set
    y_pred = trained_model.predict(X_test)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)

    # Plotting
    # Plotting
    plt.scatter(y_test, y_pred, color='black', label='Actual vs Predicted')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='blue', label='Perfect Prediction')
    plt.xlabel('Actual Sepal Length (cm)')
    plt.ylabel('Predicted Sepal Length (cm)')
    plt.title('Linear Regression Result')
    plt.legend()


    # Convert the Matplotlib plot to a base64-encoded image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data_uri = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    # Pass relevant data to the template
    coefficients = trained_model.coef_
    intercept = trained_model.intercept_

    context = {
        'coefficients': coefficients,
        'intercept': intercept,
        'plot_data_uri': f'data:image/png;base64,{plot_data_uri}',
        'mse': mse,
    }

    # Render the template with the provided context
    return render(request, 'index.html', context)


# ... (your other imports)
# views.py (Django)


def perform_knn_on_iris():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)

    # Train the model
    knn.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Visualize the confusion matrix
    plot_data_uri = plot_confusion_matrix(cm)

    return accuracy, plot_data_uri

def plot_confusion_matrix(cm):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = ['Setosa', 'Versicolor', 'Virginica']
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data_uri = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    return plot_data_uri

def knn_result(request):
    # Call your KNN function
    knn_accuracy, knn_confusion_matrix = perform_knn_on_iris()

    # Pass accuracy and plot data to the template
    context = {'knn_accuracy': knn_accuracy, 'knn_confusion_matrix': knn_confusion_matrix}

    # Render the template with the provided context
    return render(request, 'index.html', context)




def perform_logistic_regression_on_iris():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # For binary classification, let's focus on Setosa (class 0) and non-Setosa (classes 1 and 2)
    y_binary = (y == 0).astype(int)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    # Initialize the Logistic Regression classifier
    logistic_reg = LogisticRegression()

    # Train the model
    logistic_reg.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = logistic_reg.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Visualize the confusion matrix
    plot_data_uri = plot_confusion_matrix(cm)

    return accuracy, plot_data_uri

def plot_confusion_matrix(cm):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    classes = ['Not Setosa', 'Setosa']
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data_uri = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    return plot_data_uri

def logistic_regression_result(request):
    # Call your logistic regression function
    logistic_accuracy, logistic_confusion_matrix = perform_logistic_regression_on_iris()

    # Pass accuracy and plot data to the template
    context = {'logistic_accuracy': logistic_accuracy, 'logistic_confusion_matrix': logistic_confusion_matrix}

    # Render the template with the provided context
    return render(request, 'index.html', context)