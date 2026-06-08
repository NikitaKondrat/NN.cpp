#!/usr/bin/env python3
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
BUILD_DIR = BASE_DIR / 'build'
DATA_DIR = BASE_DIR / 'data' / 'prepared'
sys.path.insert(0, str(BUILD_DIR))

try:
    import bind as nn
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def compute_f1(predictions: list[float], targets: list[float], threshold: float = 0.5) -> float:
    tp = fp = fn = tn = 0
    for pred, true in zip(predictions, targets):
        pred_cls = 1 if pred >= threshold else 0
        true_cls = 1 if true >= 0.5 else 0 
        
        if pred_cls == 1 and true_cls == 1: 
            tp += 1
        if pred_cls == 1 and true_cls == 0: 
            fp += 1
        if pred_cls == 0 and true_cls == 1:
            fn += 1
        if pred_cls == 0 and true_cls == 0: 
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def load_data(filepath: str) -> tuple[list[list[float]], list[float]]:
    inputs = []
    targets = []
    with open(filepath, 'r') as f:
        header = f.readline().split()
        in_size = int(header[1])
        # out_size = int(header[2])
        
        for line in f:
            vals = list(map(float, line.split()))
            inputs.append(vals[:in_size])
            targets.append(vals[in_size])
    return inputs, targets

def train_and_evaluate(dataset_name: str, train_path: str, test_path: str, config: dict) -> float:
    dv_train = nn.FileDataVendor(train_path)
    wv = nn.RandomWeightVendor(
        n_layers=config['n_layers'],
        in_size=dv_train.in_size(),
        l_size=config['hidden_size'],
        out_size=dv_train.out_size(),
        with_bias=config['with_bias']
    )
    av = nn.ActivationVendor(config['n_layers'])
    av.set_hid(nn.Activation(nn.relu, nn.relu_deriv)) \
      .set_out(nn.Activation(nn.sigmoid, nn.sigmoid_deriv))

    nw = nn.Network(wv, av, dv_train)
    nw.set_wb(config['with_bias']).set_lp(nn.bce_lp).set_lr(config['lr'])

    print(f"Training...\n{config}\n")
    nw.epochs(config['epochs'])

    test_inputs, test_targets = load_data(test_path)
    predictions = [nw.compute(x)[0] for x in test_inputs]
    f1 = compute_f1(predictions, test_targets)
    return f1

def main():
    datasets = {
        'd1': {'train': str(DATA_DIR / "dataset1_train.txt"), 'test': str(DATA_DIR / "dataset1_test.txt")},
        'd2': {'train': str(DATA_DIR / "dataset2_train.txt"), 'test': str(DATA_DIR / "dataset2_test.txt")}
    }

    config = {
        'n_layers': 4,
        'hidden_size': 3,
        'epochs': 1000,
        'lr': 0.01,
        'with_bias': True
    }

    f1_scores = {}
    for name, paths in datasets.items():
        f1_scores[name] = train_and_evaluate(name, paths['train'], paths['test'], config)

    score = 0.5 * f1_scores['d1'] + 0.5 * f1_scores['d2']
    
    print("Results:")
    print(f"F1(d1) = {f1_scores['d1']:.4f}")
    print(f"F1(d2) = {f1_scores['d2']:.4f}")
    print(f"Score  = {score:.4f}")
        
if __name__ == "__main__":
    main()