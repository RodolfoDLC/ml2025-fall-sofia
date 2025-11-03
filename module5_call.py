#!/usr/bin/env python3

"""
Module 5 - Main Program:
Uses the NumberCollection class from module5_mod.py to process user input.
"""

from module5_mod import NumberCollection


def main():
    """Main function to run the program."""
    try:
        N = int(input("Enter the number of integers (N): ").strip())
    except Exception:
        print("Invalid input. Please enter a valid integer.")
        return
    
    if N <= 0:
        print("N must be a positive integer.")
        return
    
    # Create an instance of NumberCollection
    collection = NumberCollection()
    
    # Read N numbers one by one
    print(f"Enter {N} integers (one per line):")
    for i in range(N):
        try:
            number = int(input(f"Number {i+1}: ").strip())
            collection.add_number(number)
        except Exception:
            print("Invalid input. Please enter a valid integer.")
            return
    
    # Read the number to search for
    try:
        X = int(input("Enter the number to search for (X): ").strip())
    except Exception:
        print("Invalid input. Please enter a valid integer.")
        return
    
    # Search for X and print the result
    result = collection.search(X)
    print(f"Result: {result}")


if __name__ == "__main__":
    main()

