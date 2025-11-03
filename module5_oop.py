#!/usr/bin/env python3

"""
Module 5 - OOP version:
- Read N (positive integer)
- Read N numbers (one by one)
- Read X and print the 1-based index of X among the numbers or -1 if not found
Using Object-Oriented Programming Paradigm
"""


class NumberCollection:
    """Class to handle data initialization, insertion, and search operations."""
    
    def __init__(self):
        """Initialize an empty number collection."""
        self.numbers = []
    
    def add_number(self, number):
        """Add a number to the collection.
        
        Args:
            number (int): The number to add to the collection
        """
        self.numbers.append(number)
    
    def search(self, target):
        """Search for a target number in the collection.
        
        Args:
            target (int): The number to search for
            
        Returns:
            int: 1-based index if found, -1 if not found
        """
        try:
            # Find the index (0-based) and convert to 1-based
            index = self.numbers.index(target) + 1
            return index
        except ValueError:
            return -1


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

