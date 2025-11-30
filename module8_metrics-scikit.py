#!/usr/bin/env python3

"""
Module 8 - Precision and Recall Metrics using Scikit-learn:
- Read N (positive integer) - number of data points
- Read N (x, y) points (one by one) where x is ground truth (0 or 1) and y is predicted (0 or 1)
- Output Precision and Recall based on the inputs
Using NumPy library for data processing and Scikit-learn for ML computation
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score


class MetricsCalculator:
    """Class to handle data initialization, insertion, and metric calculation using Scikit-learn."""
    
    def __init__(self):
        """Initialize an empty point collection."""
        self.y_true = None  # Will store ground truth labels as numpy array
        self.y_pred = None  # Will store predicted labels as numpy array
        self.n_points = 0
    
    def add_points(self, y_true_array, y_pred_array):
        """Add points to the collection.
        
        Args:
            y_true_array (numpy.ndarray): Array of shape (N,) containing ground truth labels (0 or 1)
            y_pred_array (numpy.ndarray): Array of shape (N,) containing predicted labels (0 or 1)
        """
        self.y_true = y_true_array
        self.y_pred = y_pred_array
        self.n_points = len(y_true_array)
    
    def calculate_precision(self):
        """Calculate Precision using Scikit-learn.
        
        Returns:
            float: Precision score
        """
        if self.y_true is None or self.y_pred is None or self.n_points == 0:
            raise ValueError("No data available for calculation")
        
        # Calculate precision using scikit-learn
        # zero_division='warn' is the default, but we can handle it explicitly
        precision = precision_score(self.y_true, self.y_pred, zero_division=0)
        
        return precision
    
    def calculate_recall(self):
        """Calculate Recall using Scikit-learn.
        
        Returns:
            float: Recall score
        """
        if self.y_true is None or self.y_pred is None or self.n_points == 0:
            raise ValueError("No data available for calculation")
        
        # Calculate recall using scikit-learn
        # zero_division='warn' is the default, but we can handle it explicitly
        recall = recall_score(self.y_true, self.y_pred, zero_division=0)
        
        return recall


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
    
    # Read N (x, y) points
    print(f"\nEnter {N} data points (x, y) one by one:")
    print("Note: x is ground truth (0 or 1), y is predicted (0 or 1)")
    # Pre-allocate NumPy arrays for better performance
    y_true_array = np.empty(N, dtype=np.int32)
    y_pred_array = np.empty(N, dtype=np.int32)
    
    for i in range(N):
        try:
            x = int(input(f"Point {i+1} - Enter x value (ground truth, 0 or 1): ").strip())
            y = int(input(f"Point {i+1} - Enter y value (predicted, 0 or 1): ").strip())
            
            # Validate that x and y are either 0 or 1
            if x not in [0, 1]:
                print(f"Error: x value must be 0 or 1. Got {x}.")
                return
            if y not in [0, 1]:
                print(f"Error: y value must be 0 or 1. Got {y}.")
                return
            
            # Direct insertion into pre-allocated NumPy arrays
            y_true_array[i] = x
            y_pred_array[i] = y
        except ValueError:
            print("Error: Invalid input. Please enter valid integers (0 or 1) for x and y.")
            return
    
    # Create MetricsCalculator instance and add points
    metrics_calculator = MetricsCalculator()
    metrics_calculator.add_points(y_true_array, y_pred_array)
    
    # Calculate Precision and Recall using Scikit-learn
    try:
        precision = metrics_calculator.calculate_precision()
        recall = metrics_calculator.calculate_recall()
        
        print(f"\nPrecision: {precision:.6f}")
        print(f"Recall: {recall:.6f}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()

