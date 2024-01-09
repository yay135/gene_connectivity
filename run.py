import subprocess
import argparse
import os

parser = argparse.ArgumentParser(description='Run gene connectivity with multi-model multi-edge, a commandline tool')

# can be 'gcn' or 'pna' or 'linear_regression' or 'auto_encoder' or 'mlp'
model_choices=['gcn', 'pna', 'linear_regression', 'auto_encoder', 'mlp']
parser.add_argument('-m', '--model', type=str, choices=model_choices, required=True,\
                    help='specify the model type you wish to run.')
data_choices=['gtex_tcga_normal', 'tcga_ccle_bc', 'tcga_cptac_bc']
parser.add_argument('-d', '--data', type=str, choices=data_choices, required=True, help='select a dataset configuration.')
edge_choices = ["cor", "string", "dorothea", "hic_intra", "pathway", "spatial", "hic_inter"]
parser.add_argument('-e', '--edges', type=str, nargs='+', choices=edge_choices, required=False,\
                    help='choose edges can be multipe, ignored if model is not pna or gcn.')

masks = ["true", 'false']
parser.add_argument('-k', '--mask', type=str, choices=masks, required=False, default="false",\
                    help='whether to use masked dataset. default=false.')

validations = ["true", 'false']
parser.add_argument('-v', '--validation', type=str, choices=validations, required=False, default="true",\
                    help='whether to run validation after training. default=true.')


args = parser.parse_args()

# Accessing the values of arguments
model_type = args.model
fd = args.data
edges = args.edges
is_masked = True if args.mask == 'true' else False
is_validation = True if args.validation == 'true' else False

print("***************************")
print(f"model: {model_type}")
print(f"train_test_dataset: {fd}")
print(f"edges selected: {str(edges)}")
print(f"mask: {is_masked}")
print(f'validation: {is_validation}')
print("***************************")

#check integrity
model_run_files = list(filter(lambda x:x.split('.')[0]==model_type and x.split('.')[1]=='py', os.listdir('.')))
if len(model_run_files) != 1:
    print(f'can not find model run file (py) for model type {model_type}')
    exit(1)

data_folder = list(filter(lambda x:x==fd, os.listdir('.')))
if len(data_folder) != 1:
    print(f'can not find data folder {fd}')
    exit(1)


qedges = list(filter(lambda x:x in ["cor", "hic_inter", "hic_intra", "pathway", "spatial",  "string", "dorothea"], edges))
if len(qedges) < len(edges):
    print('check your edge selections')
    exit(1)

#select file to run
run_file = list(filter(lambda x:x.split('.')[0]==model_type and x.split('.')[1]=='py', os.listdir('.')))[0]
if model_type in ('gcn', 'pna'):
    if edges is None:
        print("for pna gcn, edges are required (-e/--edges)")
        exit(1)
    subprocess.run(['python', run_file, '-d', args.data, '-k', args.mask, '-v', args.validation, '-e'] + edges)
else:
    subprocess.run(['python', run_file, '-d', args.data, '-k', args.mask, '-v', args.validation])