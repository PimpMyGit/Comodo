# Comodo

• Local notebook
```
!pip install git+https://github.com/PimpMyGit/Comodo.git
```

• Colab notebook:
```
import sys
import shutil

try:
  shutil.rmtree('Comodo')
except:
  pass

!git clone https://github.com/PimpMyGit/Comodo.git
sys.path.append('/content/comodo')

!python Comodo/install_dependencies.py

from Comodo.comodo.comodo import *
```
