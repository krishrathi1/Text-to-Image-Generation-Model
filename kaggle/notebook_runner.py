# Run this inside a Kaggle notebook cell

import os
import subprocess
from pathlib import Path

REPO = "https://github.com/krishrathi1/Text-to-Image-Generation-Model.git"
ROOT = Path('/kaggle/working')

subprocess.run(['git', 'clone', REPO], check=True)
project_dir = ROOT / 'Text-to-Image-Generation-Model' / 'reckit'
os.chdir(project_dir)

subprocess.run(['python', '-m', 'pip', 'install', '-U', 'pip'], check=True)
subprocess.run(['python', '-m', 'pip', 'install', '-e', '.'], check=True)

# Single-command training launch
subprocess.run(['python', 'kaggle_train.py'], check=True)

print('Done. Checkpoints in:', project_dir / 'src' / 'diffusion' / 'models')
