import sys
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adam
from model_1.early_stop import EpochLogger, AccuracyPlateauEarlyStop
from model_1.model import CNNClassifier4

# 固定隨機種子
SEED = 27
np.random.seed(SEED)
random.seed(SEED)

detrend_methods = ["org"]

print("\n=== Model Summary (Printed Once) ===")
tmp = CNNClassifier4(input_shape=(145, 1))
print(tmp.model.summary())
print("\n=== End of Model Summary ===\n")

# ---------- 載入資料 ----------

flux_df = pd.read_csv("flux.csv")
label_df = pd.read_csv("label.csv")
param_df = pd.read_csv("param.csv")


X = flux_df.values.astype(np.float32) - 1      # shape (N, time_steps)
y = label_df['label'].values.astype(np.float32) # shape (N,)
params = param_df['param'].values
#X = X[..., np.newaxis]  # shape (N, time_steps, 1)
'''
input_root = "/data2/gigicheng/data_21/raw_data/inject_results"
input_dir = f"{input_root}/others"
X = np.loadtxt(f"{input_dir}/selected_2000_flux_data_interp_seed42.txt")-1
y = np.loadtxt(f"{input_dir}/selected_2000_labels_interp_seed42.txt")
'''
print("X", X.shape, "y", y.shape)
# ---------- 分割資料 ----------
train_idx = list(range(0, 1920))
val_idx = list(range(1920, 2160))
test_idx = list(range(2160, 2400))

X_train, Y_train = X[train_idx], y[train_idx]
X_val, Y_val = X[val_idx], y[val_idx]
X_test, Y_test = X[test_idx], y[test_idx]

# ---------- 建立 CNN ----------
cnn = CNNClassifier4(input_shape=(X.shape[1], 1))
optimizer = Adam()
cnn.model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ---------- 訓練 + 每 5 epochs 檢查錯誤 ----------
total_epochs = 40
check_every = 10
csv_logger = CSVLogger("training_log_fold1.csv", append=True)
early_stop = AccuracyPlateauEarlyStop(patience=5, threshold=0.001)

for start_epoch in range(0, total_epochs, check_every):
    end_epoch = min(start_epoch + check_every, total_epochs)
    print(f"\nTraining epochs {start_epoch+1} ~ {end_epoch}...")

    cnn.model.fit(
        X_train,
        Y_train,
        batch_size=32,
        epochs=end_epoch,
        initial_epoch=start_epoch,
        verbose=0,
        validation_data=(X_val, Y_val),
        callbacks=[csv_logger,
                   EpochLogger(interval=5, validation_data=(X_val, Y_val)),
                   early_stop]
    )

    # ---------- CNN 預測整個資料集，找出錯誤資料 ----------
    y_pred = cnn.model.predict(X_test, batch_size=32, verbose=0)
    y_pred_label = (y_pred.flatten() > 0.5).astype(int)
    wrong_idx = np.where(y_pred_label != Y_test)[0]
    
    wrong_params = params[wrong_idx]

    if len(wrong_params) > 0:
        print(f"Found {len(wrong_params)} wrong predictions after epoch {end_epoch}, saving to 'wrong_dataset.txt'")
        with open("wrong_dataset.txt", "w") as f:
            for p in wrong_params:
                f.write(p + "\n")
    else:
        print(f"All predictions correct after epoch {end_epoch}!")
    
