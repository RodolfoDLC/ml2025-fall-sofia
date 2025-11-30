import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score


def main():
    # Collect training data
    N = int(input("Enter the number of training samples (N): "))
    
    # Collect N (x, y) pairs for training set
    train_data = np.zeros((N, 2))
    print(f"Enter {N} (x, y) pairs for the training set:")
    for i in range(N):
        x = float(input(f"Enter x value for pair {i+1}: "))
        y = int(input(f"Enter y value for pair {i+1}: "))
        train_data[i, 0] = x
        train_data[i, 1] = y
    
    # Extract features and labels
    X_train = train_data[:, 0].reshape(-1, 1)  # Reshape to 2D for scikit-learn
    y_train = train_data[:, 1].astype(int)
    
    # Collect test data
    M = int(input("Enter the number of test samples (M): "))
    
    # Collect M (x, y) pairs for test set
    test_data = np.zeros((M, 2))
    print(f"Enter {M} (x, y) pairs for the test set:")
    for i in range(M):
        x = float(input(f"Enter x value for pair {i+1}: "))
        y = int(input(f"Enter y value for pair {i+1}: "))
        test_data[i, 0] = x
        test_data[i, 1] = y
    
    # Extract features and labels
    X_test = test_data[:, 0].reshape(-1, 1)  # Reshape to 2D for scikit-learn
    y_test = test_data[:, 1].astype(int)
    
    # Define kNN classifier
    knn = KNeighborsClassifier()
    
    # Determine appropriate cross-validation strategy
    # GridSearchCV uses StratifiedKFold by default for classification
    # This requires at least 5 samples in each class for 5-fold CV
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    min_class_count = np.min(class_counts)
    
    # Use KFold (non-stratified) if we can't do 5-fold stratified CV
    cv_folds = 5
    if min_class_count < 5:
        # Use KFold with appropriate number of folds
        cv_folds = min(5, N)
        cv = KFold(n_splits=cv_folds, shuffle=False)
    else:
        # Use 5-fold CV (GridSearchCV will use StratifiedKFold)
        cv = 5
    
    # Limit k values to avoid errors when k exceeds available samples in CV folds
    if isinstance(cv, KFold):
        # For KFold, each training fold has approximately N * (cv_folds-1)/cv_folds samples
        min_fold_size = max(1, int(N * (cv_folds - 1) / cv_folds) - 1)
        max_k = min(10, min_fold_size)
    else:
        # For StratifiedKFold, each training fold has approximately N * 4/5 samples
        min_fold_size = max(1, int(N * 4 / 5) - 1)
        max_k = min(10, min_fold_size)
    
    # Define parameter grid for k (1 to max_k, but at least 1 to 10 if possible)
    param_grid = {'n_neighbors': range(1, max_k + 1)}
    
    # Perform GridSearchCV with cross-validation
    grid_search = GridSearchCV(knn, param_grid, cv=cv, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Get the best k
    best_k = grid_search.best_params_['n_neighbors']
    
    # Use best estimator to predict on test set
    best_knn = grid_search.best_estimator_
    y_pred = best_knn.predict(X_test)
    
    # Calculate test accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Output results
    print(f"Best k: {best_k}")
    print(f"Test accuracy: {test_accuracy}")


if __name__ == "__main__":
    main()

