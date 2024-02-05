import os
import sys
import json


def write_to_file(file_path, d):
    with open(file_path, 'w') as file:
        file.write(json.dumps(d))
    return True

def mute():
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

