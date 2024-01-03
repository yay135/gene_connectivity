import os,sys
import torch
import numpy as np
import pandas as pd
from mlp import CustomDataset
from mlp import DataLoader
from mlp import MLP, hidden
from sklearn.metrics import mean_absolute_error
from configure import X_val_path, y_val_path, X_val_masked_path, mlp_model_path, \
mlp_model_name, mlp_model_masked_name, mlp_y_pred_path, mlp_y_pred_path_masked

masked = False
if len(sys.argv) == 2:
    masked = int(sys.argv[1])

csv_x = X_val_path if not masked else X_val_masked_path
csv_y = y_val_path
model_save_name = mlp_model_name if not masked else mlp_model_masked_name
model_path = os.path.join(mlp_model_path, model_save_name)
pred_path = mlp_y_pred_path if not masked else mlp_y_pred_path_masked

if __name__ == '__main__':

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