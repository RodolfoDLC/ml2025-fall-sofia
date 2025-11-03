#!/usr/bin/env python3

"""
Module 5 - Class Module:
Contains the NumberCollection class for data initialization, insertion, and search operations.
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

