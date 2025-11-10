#!/usr/bin/env python3

"""
Module 6 - k-NN Regression:
- Read N (positive integer) - number of data points
- Read k (positive integer) - number of nearest neighbors
- Read N (x, y) points (one by one)
- Read X (query point)
- Output Y (predicted value using k-NN regression) if k <= N, or error message otherwise
Using NumPy library for data processing and Object-Oriented Programming
"""

import numpy as np


class KNNRegression:
    """Class to handle data initialization, insertion, and k-NN regression calculation."""
    
    def __init__(self):
        """Initialize an empty point collection."""
        self.points = None  # Will store as numpy array of shape (N, 2)
        self.n_points = 0
    
    def add_points(self, points_array):
        """Add points to the collection.
        
        Args:
            points_array (numpy.ndarray): Array of shape (N, 2) containing (x, y) points
        """
        self.points = points_array
        self.n_points = len(points_array)
    
    def predict(self, X, k):
        """Perform k-NN regression to predict Y for given X.
        
        Args:
            X (float): The query point x-coordinate
            k (int): Number of nearest neighbors to use
            
        Returns:
            float: Predicted Y value using k-NN regression
        """
        if self.points is None or self.n_points == 0:
            raise ValueError("No points available for prediction")
        
        # Extract x and y coordinates
        x_coords = self.points[:, 0]
        y_coords = self.points[:, 1]
        
        # Calculate distances from X to all x-coordinates
        distances = np.abs(x_coords - X)
        
        # Get indices of k nearest neighbors (sorted by distance)
        k_nearest_indices = np.argsort(distances)[:k]
        
        # Get y-values of k nearest neighbors
        k_nearest_y = y_coords[k_nearest_indices]
        
        # Calculate mean (average) of y-values for k-NN regression
        predicted_y = np.mean(k_nearest_y)
        
        return predicted_y


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
    points_list = []
    
    for i in range(N):
        try:
            x = float(input(f"Point {i+1} - Enter x value: ").strip())
            y = float(input(f"Point {i+1} - Enter y value: ").strip())
            points_list.append([x, y])
        except ValueError:
            print("Error: Invalid input. Please enter valid real numbers for x and y.")
            return
    
    # Convert to numpy array
    points_array = np.array(points_list, dtype=np.float64)
    
    # Create KNNRegression instance and add points
    knn_regressor = KNNRegression()
    knn_regressor.add_points(points_array)
    
    # Read X (query point)
    try:
        X = float(input("\nEnter the query point X: ").strip())
    except ValueError:
        print("Error: Invalid input. Please enter a valid real number for X.")
        return
    
    # Perform k-NN regression and get predicted Y
    try:
        Y = knn_regressor.predict(X, k)
        print(f"\nResult (Y): {Y:.6f}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()

