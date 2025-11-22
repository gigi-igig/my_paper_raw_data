from sklearn.model_selection import train_test_split
from model import CNNClassifier
import pickle
import pandas as pd
import os
from tensorflow.keras.callbacks import CSVLogger, Callback

input_root = "/data2/gigicheng/data_21/raw_data/inject_results"
save_root = "/data2/gigicheng/data_21/raw_data/CNN4_layer"
detrend_methods = ["org", "cubic_sample"]

results_summary = []

for detrend_way in detrend_methods:
    print(f"\n=== Training for detrend_way: {detrend_way} ===")
    data_dir = f"{save_root}/{detrend_way}"
    os.makedirs(data_dir, exist_ok=True)
    input_dir = f"{input_root}/{detrend_way}"

    # 載入資料
    with open(f"{input_dir}/X.pkl", "rb") as f:
        X = pickle.load(f)
    with open(f"{input_dir}/y.pkl", "rb") as f:
        y = pickle.load(f)

    X = X-1
    # 切分 Test 集
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, shuffle=True
    )

    cnn = CNNClassifier(input_shape=(X.shape[1], 1))

    # CSVLogger
    csv_logger = CSVLogger(f"{data_dir}/training_log.csv", append=True)

    # 自訂 Callback: 每 5 epoch 顯示一次
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

    # 訓練模型
    cnn.model.fit(
        X_train_val,
        Y_train_val,
        batch_size=32,
        epochs=100,
        verbose=0,  # 關閉預設進度條，改由 Callback 顯示
        validation_split=0.1,
        callbacks=[csv_logger, EpochLogger(interval=5)]
    )

    # 評估模型
    loss, accuracy = cnn.model.evaluate(X_test, Y_test, batch_size=32)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
