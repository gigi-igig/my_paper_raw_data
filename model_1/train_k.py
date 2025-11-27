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
SEED_list = [26,19,41, 43, 59,13]
#19, 43, 57

input_root = "/data2/gigicheng/data_21/raw_data/inject_results/2000_bin_10_Pto1_256_d1"
save_root = "/data2/gigicheng/data_21/raw_data/CNN4_layer/2000_bin_10_Pto1_256_d1"
detrend_methods = ["org", "cubic_sample"]

results_summary = []

# 建立 log 檔並 redirect stdout
os.makedirs(save_root, exist_ok=True)
log_path = os.path.join(save_root, "training_log.txt")
log_file = open(log_path, "w")
sys.stdout = Tee(sys.stdout, log_file)

print("\n=== Model Summary (Printed Once) ===")
tmp = CNNClassifier4(input_shape=(256, 1))
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
    
    #X = X.reshape(-1, 145, 1)
    X = X - 1
    y = np.array(y)
    N = len(X)
    idx = np.arange(N)
    
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
 

        # ----------------------------
        # 動態更換 SEED 訓練機制
        # ----------------------------
        best_model = None
        final_seed_used = None

        for SEED in SEED_list:
            print(f"\nTrying SEED = {SEED} for fold {fold_idx+1}")
            np.random.seed(SEED)
            tf.random.set_seed(SEED)
            random.seed(SEED)

            # 建立新 model
            cnn = CNNClassifier4(input_shape=(X.shape[1], 1))
            optimizer = Adam()

            cnn.model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            # === 第一階段：先訓練 10 epochs 看看 val_acc 是否達標 ===
            history = cnn.model.fit(
                X_train, Y_train,
                batch_size=32,
                epochs=10,
                verbose=0,
                validation_data=(X_val, Y_val)
            )

            val_acc_10 = history.history['val_accuracy'][-1]
            print(f"SEED {SEED} → Epoch10 val_acc = {val_acc_10:.4f}")

            # 若 val_acc 達標，就使用此 SEED 繼續完整訓練
            if val_acc_10 >= 0.55:
                print(f"SEED {SEED} PASSED threshold. Continue full training.")
                final_seed_used = SEED

                epoch_logger = EpochLogger(
                    interval=5,
                    validation_data=(X_val, Y_val),
                    csv_path=f"{save_root}/training_results_epoch.csv",
                    fold=fold_idx+1,
                    seed=final_seed_used,
                    detrend_way=detrend_way
                )
                early_stop = AccuracyPlateauEarlyStop(patience=5, threshold=0.001)

                cnn.model.fit(
                    X_train, Y_train,
                    batch_size=32,
                    epochs=100,
                    verbose=0,
                    validation_data=(X_val, Y_val),
                    callbacks=[epoch_logger, early_stop]
                )

                best_model = cnn
                break
            else:
                print(f"SEED {SEED} failed (<0.55), try next SEED...")

        # 若全部 SEED 都未通過，就用最後一個 SEED 的模型結果（保底）
        if best_model is None:
            print("All SEED failed to reach val_acc >= 0.55. Using last SEED model.")
            best_model = cnn  # 最後一個 SEED 的模型
            final_seed_used = SEED_list[-1]

        print(f"Final SEED used for fold {fold_idx+1}: {final_seed_used}")

        # === 最終測試 ===
        loss, accuracy = best_model.model.evaluate(X_test, Y_test, batch_size=32)
        print(f"Fold {fold_idx+1} - Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

        # === 儲存模型 ===
        model_path = os.path.join(data_dir, f"cnn_{detrend_way}_fold{fold_idx+1}.keras")
        best_model.model.save(model_path)

        results_summary.append({
            'detrend': detrend_way,
            'fold': fold_idx+1,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'test_loss': loss,
            'test_accuracy': accuracy,
            'final_seed_used': final_seed_used
        })

# 匯出 fold 結果
results_df = pd.DataFrame(results_summary)
results_df.to_csv(f"{save_root}/sliding_kfold_results_summary.csv", index=False)

detrend_summary = []
for detrend_way in set(r['detrend'] for r in results_summary):
    fold_accs = [r['test_accuracy'] for r in results_summary if r['detrend'] == detrend_way]

    mean_acc = np.mean(fold_accs)
    acc_std = np.std(fold_accs)

    detrend_summary.append({
        'detrend_way': detrend_way,
        'mean_test_acc': mean_acc,
        'acc_std':acc_std
    })

detrend_summary = sorted(detrend_summary, key=lambda x: x["mean_test_acc"], reverse=True)
print("\n=== Detrend Comparison ===")
for r in detrend_summary:
    print(f"{r['detrend_way']}: mean_test_acc = {r['mean_test_acc']:.4f}, acc_std = {r['acc_std']:.4f}")
