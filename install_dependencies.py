import shutil
try:
  shutil.rmtree('Comodo')
except:
  pass

# !git clone https://github.com/PimpMyGit/Comodo.git
import sys
sys.path.append('/content/comodo')

import re
import os

from os import listdir
from os.path import isfile, join

from pathlib import Path

reg_import = r'(?:\n|^)(?:import|from)\s([A-Za-z0-9]+)'

cwd = os.getcwd()
comodo_dir = join(cwd, 'Comodo')

pyfiles = [str(path) for path in Path(comodo_dir).resolve().rglob('*.py')]

def read_file(filename):
    with open(filename, encoding='utf') as f:
            return f.read()

def file_dependecies(filename):
    str_content = read_file(filename)
    return re.findall(reg_import, str_content)

def all_dependecies(directory):
    pyfiles = [str(path) for path in Path(directory).resolve().rglob('*.py')]
    all_dependecies = []
    _ = [all_dependecies.extend(file_dependecies(pyfile)) for pyfile in pyfiles]
    return list(set(all_dependecies))

def module_exists(module_name):
  try:
    try_import_str = 'import ' + module_name 
    exec(try_import_str)
  except Exception as ex:
    return False
  return True

modules = all_dependecies(comodo_dir)
modules

for module_name in list(set(modules)):
  if not module_exists(module_name):
    try:
      install_str = 'pip install ' + module_name 
      os.system(install_str)
    except Exception as ex:
      print(ex)
      print('failed with module', module_name)
    
    if module_exists(module_name):
      print(module_name, 'installed succesfully')
    else:
      print('install failed with module', module_name)
