import json 

from pathlib import Path

def write_to_file(file_path, d):
    with open(file_path, 'w') as file:
        file.write(json.dumps(d)) 
    return True

def configure_save_dir():
    raise NotImplementedError

