#!/usr/bin/env python3

"""
Module 7 - k-NN Regression using Scikit-learn:
- Read N (positive integer) - number of data points
- Read k (positive integer) - number of nearest neighbors
- Read N (x, y) points (one by one)
- Read X (query point)
- Output Y (predicted value using k-NN regression) if k <= N, or error message otherwise
- Additionally provide the variance of labels in the training dataset
Using NumPy library for data processing and Scikit-learn for ML computation
"""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor


class KNNRegressionScikit:
    """Class to handle data initialization, insertion, and k-NN regression using Scikit-learn."""
    
    def __init__(self):
        """Initialize an empty point collection."""
        self.X_train = None  # Will store x coordinates as numpy array
        self.y_train = None   # Will store y coordinates as numpy array
        self.n_points = 0
        self.model = None
    
    def add_points(self, X_array, y_array):
        """Add points to the collection.
        
        Args:
            X_array (numpy.ndarray): Array of shape (N,) containing x coordinates
            y_array (numpy.ndarray): Array of shape (N,) containing y coordinates
        """
        self.X_train = X_array
        self.y_train = y_array
        self.n_points = len(X_array)
    
    def fit(self, k):
        """Fit the k-NN regression model using Scikit-learn.
        
        Args:
            k (int): Number of nearest neighbors to use
        """
        if self.X_train is None or self.n_points == 0:
            raise ValueError("No points available for training")
        
        # Reshape X_train to 2D array as required by scikit-learn
        X_train_2d = self.X_train.reshape(-1, 1)
        
        # Create and fit the k-NN regressor
        self.model = KNeighborsRegressor(n_neighbors=k)
        self.model.fit(X_train_2d, self.y_train)
    
    def predict(self, X):
        """Perform k-NN regression to predict Y for given X using Scikit-learn.
        
        Args:
            X (float): The query point x-coordinate
            
        Returns:
            float: Predicted Y value using k-NN regression
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        
        # Reshape X to 2D array as required by scikit-learn
        X_2d = np.array([[X]])
        
        # Predict using the fitted model
        predicted_y = self.model.predict(X_2d)[0]
        
        return predicted_y
    
    def get_variance(self):
        """Calculate the variance of labels (y-values) in the training dataset.
        
        Returns:
            float: Variance of the y-values
        """
        if self.y_train is None or self.n_points == 0:
            raise ValueError("No training data available")
        
        # Calculate variance using NumPy
        variance = np.var(self.y_train)
        
        return variance


def main():
    """Main function to run the program."""
    # Read N (number of points)
    try:
        N = int(input("Enter the number of data points (N): ").strip())
    except ValueError:
        print("Error: Invalid input. Please enter a valid positive integer for N.")
        return
    
    if N <= 0:
        print("Error: N must be a positive integer.")
        return
    
    # Read k (number of neighbors)
    try:
        k = int(input("Enter the number of nearest neighbors (k): ").strip())
    except ValueError:
        print("Error: Invalid input. Please enter a valid positive integer for k.")
        return
    
    if k <= 0:
        print("Error: k must be a positive integer.")
        return
    
    # Check if k <= N
    if k > N:
        print(f"Error: k ({k}) must be less than or equal to N ({N}).")
        return
    
    # Read N (x, y) points
    print(f"\nEnter {N} data points (x, y) one by one:")
    x_list = []
    y_list = []
    
    for i in range(N):
        try:
            x = float(input(f"Point {i+1} - Enter x value: ").strip())
            y = float(input(f"Point {i+1} - Enter y value: ").strip())
            x_list.append(x)
            y_list.append(y)
        except ValueError:
            print("Error: Invalid input. Please enter valid real numbers for x and y.")
            return
    
    # Convert to numpy arrays using NumPy for data processing
    X_array = np.array(x_list, dtype=np.float64)
    y_array = np.array(y_list, dtype=np.float64)
    
    # Create KNNRegressionScikit instance and add points
    knn_regressor = KNNRegressionScikit()
    knn_regressor.add_points(X_array, y_array)
    
    # Calculate and display variance of labels
    try:
        variance = knn_regressor.get_variance()
        print(f"\nVariance of labels in training dataset: {variance:.6f}")
    except Exception as e:
        print(f"Error calculating variance: {str(e)}")
        return
    
    # Fit the model
    try:
        knn_regressor.fit(k)
    except Exception as e:
        print(f"Error fitting model: {str(e)}")
        return
    
    # Read X (query point)
    try:
        X = float(input("\nEnter the query point X: ").strip())
    except ValueError:
        print("Error: Invalid input. Please enter a valid real number for X.")
        return
    
    # Perform k-NN regression and get predicted Y using Scikit-learn
    try:
        Y = knn_regressor.predict(X)
        print(f"\nResult (Y): {Y:.6f}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()

