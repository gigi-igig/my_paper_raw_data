import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.callbacks import Callback, CSVLogger
from model import CNNClassifier

save_root = "/data2/gigicheng/data_21/raw_data/inject_results"
detrend_methods = ["median_filter", "savgol", "polyfit"]

class StableAccuracyCheckpoint(Callback):
    """
    儲存模型權重，並記錄達到穩定時的各種數值
    """
    def __init__(self, filepath, stable_epochs=5, threshold=0.001):
        super().__init__()
        self.filepath = filepath
        self.stable_epochs = stable_epochs
        self.threshold = threshold
        self.val_acc_history = []
        self.stable_epochs_info = []  # list of dicts

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get("val_accuracy")
        if val_acc is None:
            return

        self.val_acc_history.append(val_acc)
        if len(self.val_acc_history) >= self.stable_epochs:
            recent = self.val_acc_history[-self.stable_epochs:]
            if max(recent) - min(recent) < self.threshold:
                print(f"Stable accuracy reached at epoch {epoch+1}: {val_acc:.4f}, saving model to {self.filepath}")
                self.model.save_weights(self.filepath)
                # 記錄各種數值
                self.stable_epochs_info.append({
                    "epoch": epoch + 1,
                    "loss": logs.get("loss"),
                    "accuracy": logs.get("accuracy"),
                    "val_loss": logs.get("val_loss"),
                    "val_accuracy": val_acc
                })
                # 避免重複記錄
                self.val_acc_history = []

# 訓練每個 detrend_way
results_summary = []

for detrend_way in detrend_methods:
    print(f"\n=== Training for detrend_way: {detrend_way} ===")
    data_dir = f"{save_root}/{detrend_way}/data"

    with open(f"{data_dir}/X_{detrend_way}.pkl", "rb") as f:
        X = pickle.load(f)
    with open(f"{data_dir}/y_{detrend_way}.pkl", "rb") as f:
        y = pickle.load(f)

    # 切分 Test
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, shuffle=True
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_idx = 1
    fold_accs = []

    all_stable_info = []  # 每個 fold 達到穩定 epoch 資訊

    for train_idx, val_idx in kf.split(X_train_val):
        print(f"\n-- Fold {fold_idx} --")
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        Y_train, Y_val = Y_train_val[train_idx], Y_train_val[val_idx]

        cnn = CNNClassifier(input_shape=(X.shape[1], 1))

        csv_logger = CSVLogger(f"{data_dir}/training_log_fold{fold_idx}.csv", append=True)
        stable_cb = StableAccuracyCheckpoint(
            filepath=f"{data_dir}/cnn_fold{fold_idx}_stable.h5",
            stable_epochs=5,
            threshold=0.001
        )

        cnn.model.fit(
            X_train, Y_train,
            batch_size=32,
            epochs=100,
            validation_data=(X_val, Y_val),
            verbose=0,
            callbacks=[csv_logger, stable_cb]
        )

        # 評估 Test
        loss, acc = cnn.model.evaluate(X_test, Y_test, batch_size=32)
        print(f"Fold {fold_idx} - Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
        fold_accs.append(acc)

        # 保存達到穩定的 epoch 資訊
        for info in stable_cb.stable_epochs_info:
            info["fold"] = fold_idx
            info["detrend_way"] = detrend_way
            all_stable_info.append(info)

        fold_idx += 1

    # 記錄每種 detrend 平均測試精度
    mean_acc = np.mean(fold_accs)
    print(f"\n=== Detrend {detrend_way} - Mean Test Accuracy: {mean_acc:.4f} ===\n")
    results_summary.append({
        "detrend_way": detrend_way,
        "mean_test_acc": mean_acc
    })

    # 保存穩定 epoch 資訊到 CSV
    df_stable = pd.DataFrame(all_stable_info)
    df_stable.to_csv(f"{data_dir}/stable_epochs_info.csv", index=False)

# 比較不同 detrend
results_summary = sorted(results_summary, key=lambda x: x["mean_test_acc"], reverse=True)
print("\n=== Detrend Comparison ===")
for r in results_summary:
    print(f"{r['detrend_way']}: mean_test_acc = {r['mean_test_acc']:.4f}")
