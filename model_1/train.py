import sys
import os
import pickle
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from model import CNNClassifier2
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adam
from early_stop import EpochLogger, AccuracyPlateauEarlyStop
from tool import Tee

# 固定隨機種子
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

input_root = "/data2/gigicheng/data_21/raw_data/inject_results/bin_10"
save_root = "/data2/gigicheng/data_21/raw_data/CNN4_layer/bin_10"
detrend_methods = ["org", "cubic_sample"]

results_summary = []

# 建立 log 檔並 redirect stdout
os.makedirs(save_root, exist_ok=True)
log_path = os.path.join(save_root, "training_log.txt")
log_file = open(log_path, "w")
sys.stdout = Tee(sys.stdout, log_file)

print("\n=== Model Summary (Printed Once) ===")
tmp = CNNClassifier2(input_shape=(145, 1))
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

    X = X - 1
    X = np.expand_dims(X, axis=-1)
    y = np.array(y)

    N = len(X)
    fold_size = N // 10
    val_size = fold_size

    for fold_idx in range(5):
        test_start = fold_idx * fold_size
        test_end = test_start + fold_size
        val_start = test_end
        val_end = val_start + val_size

        train_idx = list(range(0, test_start)) + list(range(val_end, N))
        test_idx = list(range(test_start, test_end))
        val_idx = list(range(val_start, val_end))

        X_train, Y_train = X[train_idx], y[train_idx]
        X_val, Y_val = X[val_idx], y[val_idx]
        X_test, Y_test = X[test_idx], y[test_idx]

        print(f"\n--- Fold {fold_idx+1} ---")
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        np.random.seed(SEED)
        tf.random.set_seed(SEED)
        random.seed(SEED)

        cnn = CNNClassifier2(input_shape=(X.shape[1], 1))
        optimizer = Adam(learning_rate=1e-3)

        cnn.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        csv_logger = CSVLogger(f"{data_dir}/training_log_fold{fold_idx+1}.csv", append=True)
        early_stop = AccuracyPlateauEarlyStop(patience=5, threshold=0.001)

        cnn.model.fit(
            X_train,
            Y_train,
            batch_size=32,
            epochs=300,
            verbose=0,
            validation_data=(X_val, Y_val),
            callbacks=[csv_logger, EpochLogger(interval=5), early_stop]
        )
        stopped_epoch = early_stop.stopped_epoch if early_stop.stopped_epoch is not None else 300
        loss, accuracy = cnn.model.evaluate(X_test, Y_test, batch_size=32)
        print(f"Fold {fold_idx+1} - Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

        model_path = os.path.join(data_dir, f"cnn_{detrend_way}_fold{fold_idx+1}.keras")
        cnn.model.save(model_path)

        results_summary.append({
            'detrend': detrend_way,
            'fold': fold_idx+1,
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

    detrend_summary.append({
        'detrend_way': detrend_way,
        'mean_test_acc': mean_acc,
        'mean_stopped_epoch': mean_stop
    })

detrend_summary = sorted(detrend_summary, key=lambda x: x["mean_test_acc"], reverse=True)
print("\n=== Detrend Comparison ===")
for r in detrend_summary:
    print(f"{r['detrend_way']}: mean_test_acc = {r['mean_test_acc']:.4f}")
