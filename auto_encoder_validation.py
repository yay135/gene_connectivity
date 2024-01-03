import sys
import os
import torch
import numpy as np
import pandas as pd
from auto_encoder import CustomDataset
from auto_encoder import DataLoader
from auto_encoder import RegressionAutoencoder, encoding_dim
from sklearn.metrics import mean_absolute_error
from configure import X_val_path, X_val_masked_path, y_val_path, autoenc_model_path,\
autoenc_model_name, autoenc_model_masked_name, autoenc_y_pred_path, autoenc_y_pred_path_masked

masked = False
if len(sys.argv) == 2:
    masked = int(sys.argv[1])

csv_x = X_val_path
if masked:
    csv_x = X_val_masked_path

csv_y = y_val_path
# the true values of input without masking.
csv_x_true = X_val_path
model_path = os.path.join(autoenc_model_path, autoenc_model_name)
if masked:
    model_path = os.path.join(autoenc_model_path, autoenc_model_masked_name)

pred_path = autoenc_y_pred_path
if masked:
    pred_path = autoenc_y_pred_path_masked

if __name__ == '__main__':

    custom_dataset = CustomDataset(csv_x=csv_x, csv_y=csv_y, csv_x_true=csv_x_true)
    data_loader = DataLoader(custom_dataset, batch_size=1, shuffle=False, num_workers=1)
    input_dim = custom_dataset.x.shape[1]
    out_dim = custom_dataset.y.shape[1]
    nsamples = len(custom_dataset.x)

    model = RegressionAutoencoder(input_dim=input_dim, encoding_dim=encoding_dim, out_dim=out_dim)
    model.load_state_dict(torch.load(model_path))


    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            input, _ ,out = data
            _, y_pred = model(input)
            y_pred_list.append(y_pred.numpy())
            print(f'mae: {mean_absolute_error(y_pred.numpy(), out.numpy())}')

    pred = np.concatenate(y_pred_list, axis=0)

    # caculate columns wise pcc
    assert(pred.shape == custom_dataset.y.shape)
    pred_df = pd.DataFrame(pred, columns=custom_dataset.y.columns)
    
    pccs = pred_df.corrwith(custom_dataset.y)
    pred_df.to_csv(pred_path, index=False)