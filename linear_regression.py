from configure import X_path, y_path, X_val_path, y_val_path, X_masked_path, X_val_masked_path,\
lm_y_pred_path, lm_y_pred_path_masked
from sklearn.linear_model import LinearRegression
import pandas as pd
import sys

masked = False
if len(sys.argv) == 2:
    masked = int(sys.argv[1])

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

