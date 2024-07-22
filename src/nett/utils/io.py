"""
io

This module contains utility functions for input/output operations

Functions:
    write_to_file: Write a dictionary to a file
    mute: Mute the standard output and standard error
"""
import os
import sys
import json

def write_to_file(file_path, d) -> bool:
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

def mute() -> None:
    """
    mute

    Mute the standard output and standard error
    """
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w") # TODO Should we be suppressing error messages?
