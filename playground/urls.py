from django.urls import path
from .views import linear_regression_result, knn_result,logistic_regression_result  # Import the knn_result view

urlpatterns = [
    # Add your existing paths
    # path('hello/', views.say_hello),

    # Add the path for linear regression result
    path('linear_regression_result/', linear_regression_result, name='linear_regression_result'),
     path('logistic_regression_result/', logistic_regression_result, name='logistic_regression_result'),
    # Add the path for KNN result
    path('knn_result/', knn_result, name='knn_result'),
]
