import os
import torch
import time
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset, random_split
import torch.nn.functional as F
# from configure import X_path, y_path, X_masked_path, mlp_model_path, mlp_model_name, \
#     mlp_model_masked_name

parser = argparse.ArgumentParser()

masks = ["true", 'false']
parser.add_argument('-k', '--mask', type=str, choices=masks, required=False, default="false",\
                    help='whether to use masked dataset. default=false.')

validations = ["true", 'false']
parser.add_argument('-v', '--validation', type=str, choices=validations, required=False, default="true",\
                    help='whether to run validation after training. default=true.')

data_choices=['gtex_tcga_normal', 'tcga_ccle_bc', 'tcga_cptac_bc']
parser.add_argument('-d', '--data', type=str, choices=data_choices, required=True, help='select a dataset configuration.')

args = parser.parse_args()

masked = args.mask == 'true'
validation = args.validation == 'true'
fd = args.data

#configuring path
X_path = f'{fd}/X.csv'
y_path = f'{fd}/y.csv'

X_val_path = f'{fd}/X_val.csv'
y_val_path = f'{fd}/y_val.csv'

X_masked_path = f'{fd}/mask_ratio_0.7_X.csv'
X_val_masked_path = f'{fd}/mask_ratio_0.7_X_val.csv'

out_folder = 'model_out'
if not os.path.exists(out_folder) :
    os.mkdir(out_folder)

mlp_model_path = 'mlp_states'
if not os.path.exists(mlp_model_path) :
    os.mkdir(mlp_model_path)

mlp_model_name = f'{fd}.pth'
mlp_model_masked_name = f'{fd}_exp_masked.pth'

mlp_y_pred_path = f'{out_folder}/mlp_{fd}_out.csv'
mlp_y_pred_path_masked = f'{out_folder}/mlp_{fd}_exp_masked_out.csv'

csv_x = X_path if not masked else X_masked_path
csv_y = y_path
model_save_name = mlp_model_name if not masked else mlp_model_masked_name
save_model_path = os.path.join(mlp_model_path, model_save_name)

csv_x = X_val_path if not masked else X_val_masked_path
csv_y = y_val_path
model_save_name = mlp_model_name if not masked else mlp_model_masked_name
model_path = os.path.join(mlp_model_path, model_save_name)
pred_path = mlp_y_pred_path if not masked else mlp_y_pred_path_masked


learning_rate = 0.0008
num_epochs = 100
bs = 8
hidden = 2048
train = 0.9
test = 0.1
patience = 5
cuda = 'cuda' if torch.cuda.is_available() else 'cpu'


class CustomDataset(Dataset):
    def __init__(self, csv_x, csv_y):
        self.x = pd.read_csv(csv_x)
        # If you need any preprocessing, you can do it here
        self.y = pd.read_csv(csv_y)
        assert(len(self.x) == len(self.y))
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        features = self.x.iloc[index, :].values
        label = self.y.iloc[index, :].values
        # Convert to PyTorch tensors
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return features, label

# Define the autoencoder class
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

if __name__ == '__main__':

    custom_dataset = CustomDataset(csv_x=csv_x, csv_y=csv_y)

    train_dataset, test_dataset = random_split(custom_dataset, [train, 1-train])

    input_dim = custom_dataset.x.shape[1]
    out_dim = custom_dataset.y.shape[1]
    nsamples = len(custom_dataset.x)
    # Usage:
    # Initialize the model
    model = MLP(input_size=input_dim, hidden_size=hidden, output_size=out_dim).to(cuda)
    # Define loss
    criterion = nn.MSELoss()

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=4)

    best_loss = float('inf')

    start = time.time()
    no_change_count = 0

    # Training loop
    for epoch in range(num_epochs):
        print(f'at epoch {epoch} ...')
        epoch_loss = 0
        epoch_loss_avg = 0
        
        model.train()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, (input, out) in pbar:
            input = input.to(cuda)
            out = out.to(cuda)
            output = model(input)
            loss = criterion(output, out)
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_loss_avg = round(epoch_loss/(i+1), 5)
        
            info = f'epoch: {epoch}, epoch_loss_avg: {epoch_loss_avg}'
            pbar.set_description(info)

        model.eval()
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))

        testing_loss = 0
        testing_loss_avg = 0
        with torch.no_grad():
            for j, (input, out) in pbar:
                input = input.to(cuda)
                out = out.to(cuda)
                output = model(input)
                loss = criterion(output, out)
                testing_loss += loss.item()
                testing_loss_avg = round(testing_loss/(j+1), 5)
                info = f'epoch: {epoch}, epoch_testing_loss_avg: {testing_loss_avg}'
                pbar.set_description(info)
        
        if testing_loss_avg < best_loss:
                no_change_count = 0
                best_loss = testing_loss_avg
                torch.save(model.state_dict(), save_model_path)
        else:
            no_change_count += 1
            if no_change_count > patience:
                print(f'no change after {patience} epochs stopping ...')
                break

    if validation:
        custom_dataset = CustomDataset(csv_x=csv_x, csv_y=csv_y)
        data_loader = DataLoader(custom_dataset, batch_size=1, shuffle=False, num_workers=1)
        input_dim = custom_dataset.x.shape[1]
        out_dim = custom_dataset.y.shape[1]
        nsamples = len(custom_dataset.x)

        model = MLP(input_size=input_dim, hidden_size=hidden, output_size=out_dim)

        model.load_state_dict(torch.load(model_path))


        y_pred_list = []
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                input, out = data
                y_pred = model(input)
                y_pred_list.append(y_pred.numpy())
                print(f'mae: {mean_absolute_error(y_pred.numpy(), out.numpy())}')

        pred = np.concatenate(y_pred_list, axis=0)
        pred = pd.DataFrame(pred, columns=custom_dataset.y.columns)

            # caculate columns wise pcc
        assert(pred.shape == custom_dataset.y.shape)
        pred_df = pd.DataFrame(pred, columns=custom_dataset.y.columns)
        pred.to_csv(pred_path, index=False)