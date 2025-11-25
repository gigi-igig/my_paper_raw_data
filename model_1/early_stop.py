from tensorflow.keras.callbacks import CSVLogger, Callback

import csv
import os
from tensorflow.keras.callbacks import Callback

class EpochLogger(Callback):
    def __init__(self, interval=5, validation_data=None, csv_path=None, fold=None, seed=None, detrend_way=None):
        super().__init__()
        self.interval = interval
        self.validation_data = validation_data

        # CSV 位置與額外欄位
        self.csv_path = csv_path
        self.fold = fold
        self.seed = seed
        self.detrend_way = detrend_way

        # 如果必要，建立 CSV 檔與 header
        if self.csv_path is not None:
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "epoch", "loss", "acc", "val_loss", "val_acc",
                        "fold", "seed", "detrend_way"
                    ])

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval != 0:
            return
        
        loss = logs.get("loss")
        acc = logs.get("accuracy")
        val_loss = logs.get("val_loss")
        val_acc = logs.get("val_accuracy")

        print(f"Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        # Optional：顯示 validation prediction 分布
        '''
        if self.validation_data is not None:
            X_val, Y_val = self.validation_data
            preds = self.model.predict(X_val, verbose=0)
            print(f"pred_mean={preds.mean():.4f}, pred_min={preds.min():.4f}, pred_max={preds.max():.4f}")
        '''
        # === 寫入 CSV ===
        if self.csv_path is not None:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1,
                    float(loss),
                    float(acc),
                    float(val_loss),
                    float(val_acc),
                    self.fold,
                    self.seed,
                    self.detrend_way
                ])


class AccuracyPlateauEarlyStop(Callback):
    def __init__(self, patience=5, threshold=0.001):
        super().__init__()
        self.patience = patience
        self.threshold = threshold
        self.val_acc_history = []
        self.wait = 0
        self.stopped_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get("val_accuracy")

        # 前 10 epoch 或 val_acc < 0.7 時，先累積資料，但不判斷 early stop
        if len(self.val_acc_history) < 10 or val_acc < 0.7:
            self.val_acc_history.append(val_acc)
            return

        # 計算變化幅度
        diff = abs(val_acc - self.val_acc_history[-1])
        self.val_acc_history.append(val_acc)

        if diff < self.threshold:
            self.wait += 1
        else:
            self.wait = 0  # 表現明顯進步 → 重置

        # 若連續 patience 次停滯 → early stop
        if self.wait >= self.patience:
            self.stopped_epoch = epoch + 1
            print(f"\nEarly stopping triggered at epoch {self.stopped_epoch} "
                  f"(val_accuracy change < {self.threshold} for {self.patience} epochs)")
            self.model.stop_training = True

