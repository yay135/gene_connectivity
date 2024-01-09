from sklearn.linear_model import LinearRegression
import pandas as pd
import argparse
import os

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

    # linear regression path
lm_y_pred_path = f'{out_folder}/lm_{fd}_out.csv'
lm_y_pred_path_masked = f'{out_folder}/lm_{fd}_exp_masked_out.csv'

lm = LinearRegression()
Xp = X_path if not masked else X_masked_path
X = pd.read_csv(Xp)
y = pd.read_csv(y_path)

Xp_val = X_val_path if not masked else X_val_masked_path
X_val = pd.read_csv(Xp_val)
y_val = pd.read_csv(y_val_path)

lm.fit(X, y)
y_pred = lm.predict(X_val)

out_path = lm_y_pred_path if not masked else lm_y_pred_path_masked
pd.DataFrame(y_pred, columns=y_val.columns).to_csv(out_path, index=False)

