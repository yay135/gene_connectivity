import sys 
import os

#**********************************************************
# can be 'gcn' or 'pna' or 'linear_regression' or 'auto_encoder' or 'mlp'
model_type = 'auto_encoder'
# use masked exp data ot not
mask_exp = False
# can be 'gtex_tcga_normal' or 'tcga_ccle_bc' or 'tcga_cptac_bc'
fd = 'tcga_cptac_bc'
# any combination of "cor", "hic_inter", "hic_intra", "pathway", "spatial",  "string", "dorothea"
edges = ["cor", "string", "dorothea", "hic_intra", "pathway", "spatial", "hic_inter"]
#************************************************************

# check configures

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

edge_str = '_'.join(edges)

X_path = f'{fd}/X.csv'
y_path = f'{fd}/y.csv'

X_val_path = f'{fd}/X_val.csv'
y_val_path = f'{fd}/y_val.csv'

X_masked_path = f'{fd}/mask_ratio_0.7_X.csv'
X_val_masked_path = f'{fd}/mask_ratio_0.7_X_val.csv'

out_folder = 'model_out'
if not os.path.exists(out_folder) :
    os.mkdir(out_folder)

# auto encoder path
autoenc_model_path = 'auto_encoder_states'
if not os.path.exists(autoenc_model_path) :
    os.mkdir(autoenc_model_path)

autoenc_model_name = f'{fd}.pth'
autoenc_model_masked_name = f'{fd}_exp_masked.pth'

autoenc_y_pred_path = f'{out_folder}/auto_{fd}_out.csv'
autoenc_y_pred_path_masked = f'{out_folder}/auto_{fd}_exp_masked_out.csv'

# linear regression path
lm_y_pred_path = f'{out_folder}/lm_{fd}_out.csv'
lm_y_pred_path_masked = f'{out_folder}/lm_{fd}_exp_masked_out.csv'

# mlp path
mlp_model_path = 'mlp_states'
if not os.path.exists(mlp_model_path) :
    os.mkdir(mlp_model_path)

mlp_model_name = f'{fd}.pth'
mlp_model_masked_name = f'{fd}_exp_masked.pth'

mlp_y_pred_path = f'{out_folder}/mlp_{fd}_out.csv'
mlp_y_pred_path_masked = f'{out_folder}/mlp_{fd}_exp_masked_out.csv'

gcn_model_path = 'gcn_states'
if not os.path.exists(gcn_model_path) :
    os.mkdir(gcn_model_path)

gcn_model_name = f'{fd}.pth'
gcn_model_masked_name = f'{fd}_exp_masked.pth'

gcn_y_pred_path = f'{out_folder}/mlp_{fd}_{edge_str}_out.csv'
gcn_y_pred_path_masked = f'{out_folder}/mlp_{fd}_{edge_str}_exp_masked_out.csv'

pna_model_path = 'pna_states'
if not os.path.exists(pna_model_path) :
    os.mkdir(pna_model_path)

pna_model_name = f'{fd}.pth'
pna_model_masked_name = f'{fd}_exp_masked.pth'

pna_y_pred_path = f'{out_folder}/mlp_{fd}_{edge_str}_out.csv'
pna_y_pred_path_masked = f'{out_folder}/mlp_{fd}_{edge_str}_exp_masked_out.csv'

# pna path
pathway_edge_file = f'pathway_edges/pathways_{fd}.csv'
spatial_edge_file = f'1d_edges/spatial_{fd}.csv'
hic_edge_inter = f'hic_edges_inter/hic_inter_{fd}.csv'
hic_edge_intra = f'hic_edges_intra/hic_intra_{fd}.csv'

string_edge_file = f'string_edges/string_{fd}.csv'
dorothea_edge_file = f'dorothea_edges/dorothea_{fd}.csv'



