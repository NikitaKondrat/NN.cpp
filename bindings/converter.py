#!/usr/bin/env python3
import csv
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.resolve().parent / 'data'
PREPARED_DATA_DIR = DATA_DIR / 'prepared'

def prepare_datasets(csv_name, dataset_prefix, train_ratio=0.8, seed=42):
    random.seed(seed)
    
    csv_full_path = DATA_DIR / csv_name
    with open(csv_full_path, 'r') as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)
    
    feature_cols = [c for c in rows[0].keys() if c != 'target']
    in_size = len(feature_cols)
    out_size = 1
    
    random.shuffle(rows)
    split_idx = int(len(rows) * train_ratio)
    train_rows = rows[:split_idx]
    test_rows = rows[split_idx:]
    
    def write_dataset(path, data_rows):
        with open(path, 'w') as f_out:
            f_out.write(f"{len(data_rows)} {in_size} {out_size}\n")
            for row in data_rows:
                features = [f"{float(row[col]):.8f}" for col in feature_cols]
                target = f"{float(row['target']):.8f}"
                f_out.write(' '.join(features + [target]) + '\n')
    
    write_dataset(PREPARED_DATA_DIR / f"{dataset_prefix}_train.txt", train_rows)
    write_dataset(PREPARED_DATA_DIR / f"{dataset_prefix}_test.txt", test_rows)
    print(f"{csv_full_path.name}: train={len(train_rows)}, test={len(test_rows)}, features={in_size}.")

if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PREPARED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    prepare_datasets('dataset1.csv', 'dataset1')
    prepare_datasets('dataset2.csv', 'dataset2')
    print("Datasets prepared.")