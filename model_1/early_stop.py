import csv
import os
from tensorflow.keras.callbacks import Callback

class EpochLogger(Callback):
    def __init__(self, interval, validation_data, csv_path, detrend_way, fold, seed):
        super().__init__()
        self.interval = interval
        self.validation_data = validation_data
        self.csv_path = csv_path
        self.detrend_way = detrend_way
        self.fold = fold
        self.seed = seed

        # 如果 CSV 不存在 → 建立 header
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "detrend", "fold", "seed", "epoch",
                    "loss", "acc", "val_loss", "val_acc"
                ])

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        acc = logs.get('accuracy')
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_accuracy')

        # ======== 1. interval = 5 時印出 ========
        if (epoch + 1) % self.interval == 0:
            print(f"Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
            
        # ======== 2. 無論 interval，全部記錄到 CSV ========
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.detrend_way,
                self.fold,
                self.seed,
                epoch + 1,
                float(loss),
                float(acc),
                float(val_loss),
                float(val_acc)
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

