from tensorflow.keras.callbacks import CSVLogger, Callback

class EpochLogger(Callback):
    def __init__(self, interval=5):
        super().__init__()
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            loss = logs.get('loss')
            acc = logs.get('accuracy')
            val_loss = logs.get('val_loss')
            val_acc = logs.get('val_accuracy')
            print(f"Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

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

