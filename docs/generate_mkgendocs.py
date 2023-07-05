import os
from omegaconf import OmegaConf
import sys
import ruamel.yaml
from src.simulation import callback, common, agent,  env_wrapper,algorithms

curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import inspect
import pdb
pages = {
    'sources_dir': 'docs/api_docs',
    'templates_dir': None,
    'repo': 'https://github.com/buildingamind/pipeline_embodied',
    'version': 'benchmark_experiments',
    'pages': []
}

for module in [callback, common, agent,  env_wrapper,algorithms]:
    
    last_file = None
    save_old = False
    for name, item in inspect.getmembers(module):
        
        if inspect.isclass(item) and item not in [OmegaConf, os]:
            file = inspect.getfile(item).split('/')[-1]
            page = {
                'page': file.replace('.py', '.md'),
                'source': inspect.getfile(item),
                'classes': [item.__name__]
            }
            pages['pages'].append(page)
        
        if inspect.isfunction(item):
            file = inspect.getfile(item).split('/')[-1]
            if file == last_file:
                page['functions'].append(item.__name__)
                save_old = False
            else:
                save_old = True
                page = {
                    'page': file.replace('.py', '.md'),
                    'source': inspect.getfile(item),
                    'functions': [item.__name__]
                }

            if save_old:
                pages['pages'].append(page)
            last_file = file

yaml = ruamel.yaml.YAML()
yaml.indent(sequence=4, offset=2)


with open('docs/mkgendocs.yml', 'w') as f:
    yaml.dump(pages, f)
    f.close()