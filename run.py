from configure import model_type, edges
import subprocess
import os

run_file = list(filter(lambda x:x.split('.')[0]==model_type and x.split('.')[1]=='py', os.listdir('.')))[0]
if model_type in ('gcn', 'pna'):
    subprocess.run(['python', run_file] + edges)
else:
    subprocess.run(['python', run_file])