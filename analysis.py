import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


def analyze_results():
    """
    Performs subject-level analysis on model predictions.
    """
    # --- Configuration ---
    base_dir = r'.\data'
    method_file = os.path.join(base_dir, 'Method.xlsx')
    ground_truth_file = os.path.join(base_dir,r"additional-information.txt")
    output_file = os.path.join(base_dir, 'Table 2.csv')

    # --- Load Model Predictions ---
    try:
        df = pd.read_excel(method_file, sheet_name="All_Runs_Predictions", engine="openpyxl")
    except FileNotFoundError:
        print(f"Error: Could not find the prediction file at {method_file}")
        return

    # Average the scores for each subject across all 10 runs
    df_subject_avg = df.groupby("subject")["y_score"].mean().reset_index()
    df_subject_avg.rename(columns={"y_score": "avg_score"}, inplace=True)

    # Calculate predicted AHI-like score (events per hour)
    # This assumes each prediction is for a 1-minute window
    df["y_pred_binary"] = df["y_score"] > 0.5
    # Calculate average number of positive events per subject across all runs
    ahi_pred = df.groupby("subject")["y_pred_binary"].mean() * 60
    ahi_pred.name = "predicted_ahi"

    # --- Load Ground Truth AHI ---
    original = []
    try:
        with open(ground_truth_file, "r") as f:
            for line in f:
                rows = line.strip().split("\t")
                if len(rows) == 12 and rows[0].startswith("x"):
                    # Calculate original AHI: (Apnea Events / Total Minutes) * 60
                    original.append([rows[0], float(rows[3]) / float(rows[1]) * 60])
    except FileNotFoundError:
        print(f"Error: Could not find the ground truth file at {ground_truth_file}")
        return

    original_df = pd.DataFrame(original, columns=["subject", "original_ahi"]).set_index("subject")

    # --- Combine Predictions and Ground Truth ---
    all_data = pd.concat((ahi_pred, original_df, df_subject_avg.set_index("subject")), axis=1)
    all_data.dropna(inplace=True)  # Drop subjects not present in both files

    # --- Calculate Final Subject-Level Metrics ---
    # Classify subjects based on AHI > 5 events/hour threshold
    y_true_subject = (all_data["original_ahi"] > 5).astype(int)
    y_pred_subject = (all_data["predicted_ahi"] > 5).astype(int)

    C = confusion_matrix(y_true_subject, y_pred_subject, labels=(1, 0))
    TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]

    acc = 100. * (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    sn = 100. * TP / (TP + FN) if (TP + FN) > 0 else 0
    sp = 100. * TN / (TN + FP) if (TN + FP) > 0 else 0

    # AUC is calculated on the averaged probability scores, not the derived AHI
    auc = roc_auc_score(y_true_subject, all_data["avg_score"])
    corr = all_data.corr()["original_ahi"]["predicted_ahi"]

    result = [[
        "TransCNN", acc, sn, sp, auc, corr
    ]]

    print("\n" + "=" * 50)
    print("SUBJECT-LEVEL ANALYSIS RESULTS")
    print("=" * 50)
    print(f"Accuracy: {acc:.2f}%")
    print(f"Sensitivity: {sn:.2f}%")
    print(f"Specificity: {sp:.2f}%")
    print(f"AUC: {auc:.4f}")
    print(f"Correlation: {corr:.4f}")
    print("=" * 50)

    # --- Save Results to CSV ---
    np.savetxt(output_file, result, fmt="%s", delimiter=",", comments="",
               header="Method,Accuracy(%),Sensitivity(%),Specificity(%),AUC,Corr")
    print(f"Analysis complete. Results saved to {output_file}")


if __name__ == '__main__':
    analyze_results()