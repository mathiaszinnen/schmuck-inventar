import pandas as pd
import random
from jiwer import wer, cer

def word_error_rate(ref, hyp):
    return wer(ref, hyp)

def char_error_rate(ref, hyp):
    return cer(ref, hyp)

def compute_column_error_rate(csv1_path, csv2_path, mode='wer'):
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    df1 = df1.set_index('filename')
    df2 = df2.set_index('filename')
    common_files = df1.index.intersection(df2.index)
    columns = [col for col in df1.columns if col in df2.columns]
    error_rate_per_column = {}
    error_func = word_error_rate if mode == 'wer' else char_error_rate
    for col in columns:
        errors = []
        for fname in common_files:
            ref = str(df1.at[fname, col])
            hyp = str(df2.at[fname, col])
            errors.append(error_func(ref, hyp))
        error_rate_per_column[col] = sum(errors) / len(errors) if errors else None
    return error_rate_per_column

def compute_overall_error_rate(csv1_path, csv2_path, mode='wer'):
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    df1 = df1.set_index('filename')
    df2 = df2.set_index('filename')
    common_files = df1.index.intersection(df2.index)
    columns = [col for col in df1.columns if col in df2.columns]
    total_errors = 0
    total_units = 0
    error_func = word_error_rate if mode == 'wer' else char_error_rate
    for col in columns:
        for fname in common_files:
            ref = str(df1.at[fname, col])
            hyp = str(df2.at[fname, col])
            if mode == 'wer':
                units = ref.split()
            else:
                units = list(ref)
            errors = error_func(ref, hyp) * max(1, len(units))
            total_errors += errors
            total_units += len(units)
    overall_error_rate = total_errors / max(1, total_units)
    return overall_error_rate

def compute_map_cer(csv_gt_path, csv_pred_path, overlap_thresholds=[0.5]):
    df_gt = pd.read_csv(csv_gt_path).set_index('filename')
    df_pred = pd.read_csv(csv_pred_path).set_index('filename')
    common_files = df_gt.index.intersection(df_pred.index)
    columns = [col for col in df_gt.columns if col in df_pred.columns]
    results = {}

    for thresh in overlap_thresholds:
        aps = []
        for col in columns:
            tp = 0
            fp = 0
            fn = 0
            for fname in common_files:
                gt = str(df_gt.at[fname, col])
                pred = str(df_pred.at[fname, col])
                cer = char_error_rate(gt, pred)
                overlap = 1 - cer  # overlap analogous to IoU
                if len(gt) == 0 and len(pred) == 0:
                    continue  # ignore empty ground truth and prediction
                if overlap >= thresh:
                    tp += 1
                else:
                    if len(pred) > 0:
                        fp += 1
                    if len(gt) > 0:
                        fn += 1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            ap = precision  # for this setting, AP is just precision at threshold
            aps.append(ap)
        results[thresh] = sum(aps) / len(aps) if aps else None
    return results


def add_noise_to_csv(input_csv, output_csv, noise_level=0.1, seed=42):
    """
    As a sanity check, this function adds character-level noise to a CSV file.
    Adds character-level noise to each string cell in the CSV (excluding 'filename').
    noise_level: fraction of characters to randomly replace per cell.
    """
    random.seed(seed)
    df = pd.read_csv(input_csv)
    noisy_df = df.copy()
    for col in df.columns:
        if col == 'filename':
            continue
        for idx, val in df[col].items():
            s = str(val)
            chars = list(s)
            n = max(1, int(len(chars) * noise_level)) if chars else 0
            for _ in range(n):
                if chars:
                    pos = random.randint(0, len(chars) - 1)
                    chars[pos] = random.choice('abcdefghijklmnopqrstuvwxyz')
            noisy_df.at[idx, col] = ''.join(chars)
    noisy_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    csv1_path = "dataset/annotations.csv"
    csv2_path = "dataset/predictions_pero.csv"

    print("Column-wise WER:")
    print(compute_column_error_rate(csv1_path, csv2_path, mode='wer'))

    print("Column-wise CER:")
    print(compute_column_error_rate(csv1_path, csv2_path, mode='cer'))

    print("Overall WER:")
    print(compute_overall_error_rate(csv1_path, csv2_path, mode='wer'))

    print("Overall CER:")
    print(compute_overall_error_rate(csv1_path, csv2_path, mode='cer'))

    print("mAP CER (thresholds 0.5, 0.7, 0.9):")
    print(compute_map_cer(csv1_path, csv2_path, overlap_thresholds=[0.5, 0.7, 0.9]))

    # Print top 5 columns by CER
    cer_scores = compute_column_error_rate(csv1_path, csv2_path, mode='cer')
    top5_cer = sorted(cer_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 columns by CER:")
    for col, score in top5_cer:
        print(f"{col}: {score}")

    # Print top 5 columns by WER
    wer_scores = compute_column_error_rate(csv1_path, csv2_path, mode='wer')
    top5_wer = sorted(wer_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 columns by WER:")
    for col, score in top5_wer:
        print(f"{col}: {score}")

    # Print worst 5 columns by WER
    worst5_wer = sorted(wer_scores.items(), key=lambda x: x[1])[:5]
    print("Worst 5 columns by WER:")
    for col, score in worst5_wer:
        print(f"{col}: {score}")

    # Print worst 5 columns by CER
    worst5_cer = sorted(cer_scores.items(), key=lambda x: x[1])[:5]
    print("Worst 5 columns by CER:")
    for col, score in worst5_cer:
        print(f"{col}: {score}")



# #Example usage:
# add_noise_to_csv("dataset/annotations.csv", "dataset/annotations_noise.csv", noise_level=0.1)