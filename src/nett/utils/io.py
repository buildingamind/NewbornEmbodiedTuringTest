"""This module contains utility functions for input/output operations
"""
import os
import sys
import json

def write_to_file(file_path, d):
    """
    write_to_file
    
    Write a dictionary to a file
    
    Args:
        file_path (str): The path to the file
        d (dict): The dictionary to write
        
    Returns:
        bool: True if the file was written, False otherwise
    """
    with open(file_path, "w") as file:
        file.write(json.dumps(d))
    return True

def muteOutput():
    """Mute the standard output
    """
    sys.stdout = open(os.devnull, "w")

def muteAll():
    """Mute the standard error and standard output
    """
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
