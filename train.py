import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from sklearn.metrics import confusion_matrix, f1_score
from keras.backend import clear_session, get_session, set_session

# Import from local modules
from data_loader import load_data
from model import create_model


def reset_keras():
    """Resets the Keras session to free up memory."""
    sess = get_session()
    clear_session()
    sess.close()
    gc.collect()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tf.compat.v1.Session(config=config))


def lr_schedule(epoch, lr):
    """Defines the learning rate schedule."""
    if epoch >= 40 and (epoch - 1) % 10 == 0:
        lr *= 0.1
    print(f"Epoch {epoch + 1}: Learning rate is {lr:.6f}.")
    return lr


def main():
    # --- GPU Configuration ---
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU available: {gpus}")
        except RuntimeError as e:
            print(e)

    # --- Setup Directories and Files ---
    base_dir_out = r'.\data'
    os.makedirs(base_dir_out, exist_ok=True)

    writer1 = pd.ExcelWriter(os.path.join(base_dir_out, 'Result.xlsx'))
    writer2 = pd.ExcelWriter(os.path.join(base_dir_out, 'Method.xlsx'))

    # --- Training Configuration ---
    total_epochs = 70
    lr_scheduler_cb = LearningRateScheduler(lr_schedule)
    num_runs = 1

    # Lists to store metrics across all runs
    all_runs_metrics = {
        "acc": [], "sn": [], "sp": [], "f1": [],
        "TP": [], "TN": [], "FP": [], "FN": []
    }
    all_runs_predictions = []

    # --- Main Training Loop (10 runs) ---
    for j in range(num_runs):
        print("\n" + "=" * 50)
        print(f"STARTING RUN {j + 1}/{num_runs}")
        print("=" * 50)

        # Load data for each run to ensure random splits are consistent if seed is set in load_data
        x_train1, _, x_train3, y_train, _, x_val1, _, x_val3, y_val, _, \
            x_test1, _, x_test3, y_test, groups_test = load_data()

        y_train_cat = to_categorical(y_train, num_classes=2)
        y_val_cat = to_categorical(y_val, num_classes=2)
        y_test_cat = to_categorical(y_test, num_classes=2)

        reset_keras()
        model = create_model(x_train1.shape[1:], x_train3.shape[1:])
        model.summary()

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        filepath = os.path.join(base_dir_out, f'weights_{j + 1}times.keras')
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [lr_scheduler_cb, checkpoint]

        history = model.fit([x_train1, x_train3], y_train_cat, batch_size=64, epochs=total_epochs,
                            validation_data=([x_val1, x_val3], y_val_cat), callbacks=callbacks_list)

        # --- Evaluation ---
        print(f"\n--- Evaluating Run {j + 1} on Test Set ---")
        best_model = tf.keras.models.load_model(filepath)
        y_score = best_model.predict([x_test1, x_test3])

        # Store predictions for subject-level analysis
        run_preds = pd.DataFrame({"y_true": y_test, "y_score": y_score[:, 1], "subject": groups_test, "run": j + 1})
        all_runs_predictions.append(run_preds)

        y_true, y_pred = y_test, np.argmax(y_score, axis=-1)
        C = confusion_matrix(y_true, y_pred, labels=(1, 0))
        TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]

        acc = 100. * (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        sn = 100. * TP / (TP + FN) if (TP + FN) > 0 else 0
        sp = 100. * TN / (TN + FP) if (TN + FP) > 0 else 0
        f1 = 100. * f1_score(y_true, y_pred, average='binary')

        print(f"Run {j + 1} Results: acc: {acc:.2f}%, sn: {sn:.2f}%, sp: {sp:.2f}%, f1: {f1:.2f}%")

        all_runs_metrics["acc"].append(acc)
        all_runs_metrics["sn"].append(sn)
        all_runs_metrics["sp"].append(sp)
        all_runs_metrics["f1"].append(f1)
        all_runs_metrics["TP"].append(TP)
        all_runs_metrics["TN"].append(TN)
        all_runs_metrics["FP"].append(FP)
        all_runs_metrics["FN"].append(FN)

    # --- Final Aggregation and Saving ---
    pd.DataFrame(all_runs_metrics).to_excel(writer1, sheet_name="All_Runs_Metrics", index=False)
    pd.concat(all_runs_predictions).to_excel(writer2, sheet_name="All_Runs_Predictions", index=False)

    print("\n" + "=" * 50)
    print("OVERALL RESULTS (10 RUNS)")
    print("=" * 50)
    for key, values in all_runs_metrics.items():
        if key not in ["TP", "TN", "FP", "FN"]:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{key.upper()} -> Mean: {mean_val:.2f}%, Std Dev: {std_val:.2f}%")

    writer1.close()
    writer2.close()
    print("\nTraining and evaluation complete. Results saved.")


if __name__ == '__main__':
    main()