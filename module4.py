#!/usr/bin/env python3

"""
Simple program for module4:
- Read N (positive integer)
- Read N numbers (one by one)
- Read X and print the 1-based index of X among the numbers or -1 if not found
"""

def main():
    try:
        N = int(input("Enter the number of integers (N): ").strip())
    except Exception:
        print("Invalid input. Please enter a valid integer.")
        return
    if N <= 0:
        print("N must be a positive integer.")
        return
    
    nums = []
    print(f"Enter {N} integers (one per line):")
    for i in range(N):
        try:
            v = int(input(f"Number {i+1}: ").strip())
        except Exception:
            print("Invalid input. Please enter a valid integer.")
            return
        nums.append(v)
    
    try:
        X = int(input("Enter the number to search for (X): ").strip())
    except Exception:
        print("Invalid input. Please enter a valid integer.")
        return
    
    try:
        idx = nums.index(X) + 1
        print(f"Result: {idx}")
    except ValueError:
        print("Result: -1")

if __name__ == "__main__":
    main()
