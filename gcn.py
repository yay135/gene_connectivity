import os
import random
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, Linear


parser = argparse.ArgumentParser()

masks = ["true", 'false']
parser.add_argument('-k', '--mask', type=str, choices=masks, required=False, default="false",\
                    help='whether to use masked dataset. default=false.')

validations = ["true", 'false']
parser.add_argument('-v', '--validation', type=str, choices=validations, required=False, default="true",\
                    help='whether to run validation after training. default=true.')

data_choices=['gtex_tcga_normal', 'tcga_ccle_bc', 'tcga_cptac_bc']
parser.add_argument('-d', '--data', type=str, choices=data_choices, required=True, help='select a dataset configuration.')

edge_choices = ["cor", "string", "dorothea", "hic_intra", "pathway", "spatial", "hic_inter"]
parser.add_argument('-e', '--edges', type=str, nargs='+', choices=edge_choices, required=True,\
                    help='choose edges can be multipe, ignored if model is not pna or gcn.')

args = parser.parse_args()

is_masked = args.mask == 'true'
validate = args.validation == 'true'
fd = args.data
edges = args.edges
edge_str = '_'.join(edges)
choices = args.edges

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

gcn_model_path = 'gcn_states'
if not os.path.exists(gcn_model_path) :
    os.mkdir(gcn_model_path)

gcn_model_name = f'{fd}.pth'
gcn_model_masked_name = f'{fd}_exp_masked.pth'

gcn_y_pred_path = f'{out_folder}/mlp_{fd}_{edge_str}_out.csv'
gcn_y_pred_path_masked = f'{out_folder}/mlp_{fd}_{edge_str}_exp_masked_out.csv'

pathway_edge_file = f'pathway_edges/pathways_{fd}.csv'
spatial_edge_file = f'1d_edges/spatial_{fd}.csv'
hic_edge_inter = f'hic_edges_inter/hic_inter_{fd}.csv'
hic_edge_intra = f'hic_edges_intra/hic_intra_{fd}.csv'

string_edge_file = f'string_edges/string_{fd}.csv'
dorothea_edge_file = f'dorothea_edges/dorothea_{fd}.csv'

learning_rate = 0.0001
num_epochs = 100

patience = 5
cuda = 'cuda' if torch.cuda.is_available() else 'cpu'

cor_th = 0.9
train_ratio = 0.9

X_masked = pd.read_csv(X_masked_path)
X = pd.read_csv(X_path)
y = pd.read_csv(y_path)

def save_model_path():
    folder = gcn_model_path
    full = f"{folder}/{gcn_model_masked_name}" if is_masked else f"{folder}/{gcn_model_name}"
    return full
#shuffle
_index = [i for i in range(len(X))]
random.shuffle(_index)
X_masked = X_masked.iloc[_index, :]
X = X.iloc[_index, :]
y = y.iloc[_index, :]

# use combined X, y as model output for training forcing the model to reconstruct X.
z = pd.concat((X, y), axis=1)
# the input dataset to the model
z_in = X.reindex(columns=z.columns).fillna(0)
if is_masked:
    z_in = X_masked.reindex(columns=z.columns).fillna(0)

edge_index = None
edge_attr = []
edge_type = 0


def edge_add(edge_index, edge_attr, edge_type, add_index):
    if edge_index is None:
        edge_index = add_index        
    else:
        edge_index = np.concatenate((edge_index, add_index), axis=1)

    edge_attr += [edge_type] * add_index.shape[1] 

    edge_type += 1

    return edge_index, edge_attr, edge_type

if "cor" in choices:
    cor_mat = np.corrcoef(z.values, rowvar=False)
    row_id, col_id = np.where(cor_mat > cor_th)

    del cor_mat

    edge_index, edge_attr, edge_type = \
            edge_add(edge_index, edge_attr, edge_type, np.array([row_id, col_id]))
    
    print(edge_index.shape)

# include spatial edges
if "spatial" in choices:
    spatial_edges = pd.read_csv(spatial_edge_file).T.values
    edge_index, edge_attr, edge_type = \
            edge_add(edge_index, edge_attr, edge_type, spatial_edges)
    
    print(edge_index.shape)

if "pathway" in choices:
    pathway_edges = pd.read_csv(pathway_edge_file, index_col=0).T.values
    edge_index, edge_attr, edge_type = \
            edge_add(edge_index, edge_attr, edge_type, pathway_edges)
    
    print(edge_index.shape)

if "string" in choices:
    string_edges = pd.read_csv(string_edge_file).T.values
    edge_index, edge_attr, edge_type = \
            edge_add(edge_index, edge_attr, edge_type, string_edges)
    
    print(edge_index.shape)

if "dorothea" in choices:
    dorothea_edges = pd.read_csv(dorothea_edge_file).T.values
    edge_index, edge_attr, edge_type = \
            edge_add(edge_index, edge_attr, edge_type, dorothea_edges)
    
    print(edge_index.shape)
    
if "hic_intra" in choices:
    hic_edges = pd.read_csv(hic_edge_intra).T.values
    edge_index, edge_attr, edge_type = \
            edge_add(edge_index, edge_attr, edge_type, hic_edges)
    print(edge_index.shape)

if "hic_inter" in choices:
    hic_edges = pd.read_csv(hic_edge_inter).T.values
    edge_index, edge_attr, edge_type = \
            edge_add(edge_index, edge_attr, edge_type, hic_edges)
    print(edge_index.shape)


def configure_nodes(X, y, X_masked, is_masked):
    # use combined X, y as model output for training forcing the model to reconstruct X.
    z = pd.concat((X, y), axis=1)
    # the input dataset to the model
    z_in = X.reindex(columns=z.columns).fillna(0)
    if is_masked:
        z_in = X_masked.reindex(columns=z.columns).fillna(0)
    
    return z_in, z

def configure_graph_iterator(start, end, z, z_in, edge_index, edge_attr):

    def get_next_graph():
            limit = end
            counter = start
            while counter < limit:
                yield Data(x=torch.tensor(z_in.iloc[counter, :].values.reshape(-1, 1), dtype=torch.float), \
                        y=torch.tensor(z.iloc[counter, :].values, dtype=torch.float), \
                            edge_index=torch.tensor(edge_index, dtype=torch.long), \
                                edge_attr=torch.tensor(edge_attr, dtype=torch.float))
                counter += 1
    return get_next_graph

class Net(torch.nn.Module):
    def __init__(self, y_dim):
        super(Net, self).__init__()
        self.conv1 = GCNConv(1, 8)
        self.regression = Linear(-1, y_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x.flatten())
        x = self.regression(x)
        return x
    
if __name__ == '__main__':
        z_in, z = configure_nodes(X, y, X_masked, is_masked)

        idx_train = 0
        idx_test = int(train_ratio*len(X))

        length_train = idx_test
        length_test = len(X) - length_train

        y_dim = z.shape[1]

        get_next_graph_train = configure_graph_iterator(idx_train, idx_test, z, z_in, edge_index, edge_attr)

        get_next_graph_test = configure_graph_iterator(idx_test, len(X), z, z_in, edge_index, edge_attr)

        # the length of the prediction section
        model = Net(y_dim=y_dim).to(cuda)
        criterion = torch.nn.MSELoss().to(cuda)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        best_loss = float('inf')
        no_change_count = 0

        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_loss_avg = 0
            model.train()
            pbar = tqdm(enumerate(get_next_graph_train()), total=length_train)
            for i, data in pbar:
                data = data.to(cuda)
                optimizer.zero_grad() 
                out = model(data)
                loss = criterion(out, data.y)
                loss.backward()  
                optimizer.step()

                epoch_loss += loss.item()
                epoch_loss_avg = round(epoch_loss/(i+1), 5)
                info = f'is_masked: {is_masked}, epoch: {epoch}, batch: {i}, epoch_loss_avg: {epoch_loss_avg}'
                pbar.set_description(info)

            model.eval()
            pbar = tqdm(enumerate(get_next_graph_test()), total=length_test)

            testing_loss = 0
            testing_loss_avg = 0

            info = ''
            with torch.no_grad():
                for j, data in pbar:
                    data = data.to(cuda)
                    out = model(data)
                    loss = criterion(out, data.y)

                    testing_loss += loss.item()
                    testing_loss_avg = round(testing_loss/(j+1), 5)
                    info = f'epoch:{epoch}, batch:{j}, epoch_testing_loss_avg: {testing_loss_avg}'
                    pbar.set_description(info)

            # record val loss:
            with open(f'record_{is_masked}.txt', 'a') as f:
                f.write(info + '\n')

            if testing_loss_avg < best_loss:
                    no_change_count = 0
                    best_loss = testing_loss_avg
                    torch.save(model.state_dict(), save_model_path())
            else:
                no_change_count += 1
                if no_change_count > patience:
                    print(f'no change after {patience} epochs, best loss: {best_loss} stopping ...')
                    break

if validate:
    # read validation data
    X_val_masked = pd.read_csv(X_val_masked_path)
    X_val = pd.read_csv(X_val_path)
    y_val = pd.read_csv(y_val_path)

    X_shape = X_val.shape[1]

    # convert into input and output
    z_out = pd.concat((X_val, y_val), axis=1)
    z_in = X_val.reindex(columns=z_out.columns).fillna(0)

    model_path = save_model_path()
    out_path = gcn_y_pred_path_masked if is_masked else gcn_y_pred_path


    def get_next_graph():
        counter = 0
        while counter < len(z_in):
            yield Data(x=torch.tensor(z_in.iloc[counter, :].values.reshape(-1,1), dtype=torch.float), \
                    y=torch.tensor(z_out.iloc[counter, :].values, dtype=torch.float), \
                        edge_index=torch.tensor(edge_index, dtype=torch.long), \
                            edge_attr=torch.tensor(edge_attr, dtype=torch.float))
            counter += 1

    model = Net(y_dim=y_dim)
    model.load_state_dict(torch.load(model_path))
    model.to(cuda)
    model.eval()
    pbar = tqdm(enumerate(get_next_graph()), total=len(z_in))


    y_true_list, y_pred_list = [], []
    for i, data in pbar:
        data = data.to(cuda)
        out = model(data)

        # remove the x part of the output leave y only
        y_pred = out.cpu().detach().numpy()[X_shape:]

        y_pred_list.append(y_pred.reshape(1, -1))

    y_pred_mat = np.concatenate(y_pred_list, axis=0)

    print(y_pred_mat.shape)

    pd.DataFrame(y_pred_mat).to_csv(out_path, index=False)