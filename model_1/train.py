import sys
import os
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adam
from model_1.early_stop import EpochLogger, AccuracyPlateauEarlyStop
from model_1.tool import Tee
from model_1.model import CNNClassifier4

# 固定隨機種子
SEED = 52
#np.random.seed(SEED)
#tf.random.set_seed(SEED)
#random.seed(SEED)

input_root = "/data2/gigicheng/data_21/raw_data/inject_results/2000_bin_10_Pto1_256_d1"
save_root = "/data2/gigicheng/data_21/raw_data/CNN4_layer/test"
detrend_methods = ["org", "cubic_sample"]

results_summary = []

# 建立 log 檔並 redirect stdout
os.makedirs(save_root, exist_ok=True)
log_path = os.path.join(save_root, "training_log.txt")
log_file = open(log_path, "w")
sys.stdout = Tee(sys.stdout, log_file)

print("\n=== Model Summary (Printed Once) ===")
tmp = CNNClassifier4(input_shape=(145, 1))
print(tmp.model.summary())
print("\n=== End of Model Summary ===\n")

# 開始訓練
for detrend_way in detrend_methods:
    print(f"\n=== Training for detrend_way: {detrend_way} ===")
    data_dir = f"{save_root}/{detrend_way}"
    os.makedirs(data_dir, exist_ok=True)
    input_dir = f"{input_root}/{detrend_way}"

    with open(f"{input_dir}/X.pkl", "rb") as f:
        X = pickle.load(f)
    with open(f"{input_dir}/y.pkl", "rb") as f:
        y = pickle.load(f)
    
    #X = X.reshape(-1, 256, 1)
    X = X - 1
    y = np.array(y)
    print("X", X.shape, "y", y.shape)


    train_idx = list(range(0, 9600))
    test_idx = list(range(9600, 10800))
    val_idx = list(range(10800, 12000))

    X_train, Y_train = X[train_idx], y[train_idx]
    X_val, Y_val = X[val_idx], y[val_idx]
    X_test, Y_test = X[test_idx], y[test_idx]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print("Y_train mean:", np.mean(Y_train))
    print("Y_val mean:", np.mean(Y_val))
    print("Y_test mean:", np.mean(Y_test))


    cnn = CNNClassifier4(input_shape=(X.shape[1], 1))
    optimizer = Adam()

    cnn.model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    csv_logger = CSVLogger(f"{data_dir}/training_log_fold.csv", append=True)
    early_stop = AccuracyPlateauEarlyStop(patience=5, threshold=0.001)

    cnn.model.fit(
            X_train,
            Y_train,
            batch_size=32,
            epochs=100,
            verbose=0,
            validation_data=(X_val, Y_val),
            callbacks=[csv_logger, EpochLogger(interval=5), early_stop]
        )
    stopped_epoch = early_stop.stopped_epoch if early_stop.stopped_epoch is not None else 300
    loss, accuracy = cnn.model.evaluate(X_test, Y_test, batch_size=32)
    print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

    model_path = os.path.join(data_dir, f"cnn_{detrend_way}_fold.keras")
    cnn.model.save(model_path)

    results_summary.append({
            'detrend': detrend_way,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'test_loss': loss,
            'test_accuracy': accuracy,
            'stopped_epoch': stopped_epoch
    })

# 匯出 fold 結果
results_df = pd.DataFrame(results_summary)
results_df.to_csv(f"{save_root}/sliding_kfold_results_summary.csv", index=False)

detrend_summary = []
for detrend_way in set(r['detrend'] for r in results_summary):
    fold_accs = [r['test_accuracy'] for r in results_summary if r['detrend'] == detrend_way]
    fold_stops = [r['stopped_epoch'] for r in results_summary if r['detrend'] == detrend_way]

    mean_acc = np.mean(fold_accs)
    mean_stop = np.mean(fold_stops)
    acc_std = np.std(fold_accs)

    detrend_summary.append({
        'detrend_way': detrend_way,
        'mean_test_acc': mean_acc,
        'mean_stopped_epoch': mean_stop,
        'acc_std':acc_std
    })

detrend_summary = sorted(detrend_summary, key=lambda x: x["mean_test_acc"], reverse=True)
print("\n=== Detrend Comparison ===")
for r in detrend_summary:
    print(f"{r['detrend_way']}: mean_test_acc = {r['mean_test_acc']:.4f}, acc_std = {r['acc_std']:.4f}")
