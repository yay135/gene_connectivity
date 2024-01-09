from configure import model_type, edges, fd
import subprocess
import os

print("***************************")
print(f"model: {model_type}")
print(f"train_test_dataset: {fd}")
if model_type in ('gcn', 'pna'):
    print(f"edges selected: {str(edges)}")
print("***************************")


run_file = list(filter(lambda x:x.split('.')[0]==model_type and x.split('.')[1]=='py', os.listdir('.')))[0]
if model_type in ('gcn', 'pna'):
    subprocess.run(['python', run_file] + edges)
else:
    subprocess.run(['python', run_file])