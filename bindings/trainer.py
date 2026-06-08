#!/usr/bin/env python3
import sys
from pathlib import Path
import argparse

BASE_DIR = Path(__file__).parent.parent
BUILD_DIR = BASE_DIR / 'build'
DATA_DIR = BASE_DIR / 'data'
PREDARED_DATA_DIR = DATA_DIR / 'prepared'
WEIGHTS_DIR = DATA_DIR / 'weights'
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

def load_data(filepath):
    inputs, targets = [], []
    with open(filepath, 'r') as f:
        header = f.readline().split()
        in_size = int(header[1])
        for line in f:
            vals = list(map(float, line.split()))
            inputs.append(vals[:in_size])
            targets.append(vals[in_size])
    return inputs, targets

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate neural network")
    parser.add_argument('--train', default='', help="CSV train dataset file prefix (e.g., 'dataset1')")
    parser.add_argument('--test', default='', help="CSV test dataset file prefix (e.g., 'dataset1')")
    parser.add_argument('--weights', default=None, help="File name to saved weights to load before training")
    parser.add_argument('--save-weights', default=None, help="File name to save weights after training")
    parser.add_argument('--n-layers', type=int, default=4, help="Number of layers (default: 4)")
    parser.add_argument('--hidden-size', type=int, default=3, help="Hidden layer size (default: 3)")
    parser.add_argument('--epochs', type=int, default=1000, help="Number of epochs (default: 1000)")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument('--no-bias', action='store_true', default=False, help="Disable bias")
    parser.add_argument('--loss', choices=['bce', 'mse'], default='bce', help="Loss function (default: bce)")
    args = parser.parse_args()
    
    train_path = str(PREDARED_DATA_DIR / f"{args.train}_train.txt")
    test_path = str(PREDARED_DATA_DIR / f"{args.test}_test.txt")
    
    if not Path(train_path).exists() and not Path(test_path).exists():
        print(f"Error: Dataset files not found for")
        print(f"Expected: 'train' or 'test' dataset")
        sys.exit(1)
    
    if args.train:
        dv_train = nn.FileDataVendor(train_path)
        in_size = dv_train.in_size()
        out_size = dv_train.out_size()

    if args.test:
        dv_test = nn.FileDataVendor(test_path)
        in_size = dv_test.in_size()
        out_size = dv_test.out_size()

    if args.weights:
        print(f"Loading weights from {WEIGHTS_DIR / args.weights}...")
        wv = nn.FileWeightVendor(str(WEIGHTS_DIR / args.weights))
    else:
        print("Initializing random weights..")
        wv = nn.RandomWeightVendor(
            n_layers=args.n_layers,
            in_size=in_size,
            l_size=args.hidden_size,
            out_size=out_size,
            with_bias=not args.no_bias
        )
    av = nn.ActivationVendor(args.n_layers)
    av.set_hid(nn.Activation(nn.relu, nn.relu_deriv)) \
      .set_out(nn.Activation(nn.sigmoid, nn.sigmoid_deriv))
    
    nw = nn.Network(wv, av, dv_train if args.train else None)
    nw.set_wb(not args.no_bias)
    nw.set_lr(args.lr)
    if args.loss == 'bce':
        nw.set_lp(nn.bce_lp)
    if args.loss == 'mse':
        nw.set_lp(nn.mse_lp)
    
    if args.train:
        print(f"Training {args.train}...")
        print(f"Layers: {args.n_layers}, InputSize: {in_size}, HiddenSize: {args.hidden_size}, OutputSize: {out_size}, Epochs: {args.epochs}, LR: {args.lr}, Bias: {not args.no_bias}")
        nw.epochs(args.epochs)
    
    if args.test:
        test_inputs, test_targets = load_data(test_path)
        predictions = [nw.compute(x)[0] for x in test_inputs]
        f1 = compute_f1(predictions, test_targets)
        print(f"\nResults for {args.test}:")
        print(f"F1 = {f1:.4f}")
    
    if args.save_weights:
        WEIGHTS_DIR.mkdir(exist_ok=True, parents=True)
        print(f"Saving weights to {WEIGHTS_DIR / args.save_weights}...")
        nn.save_weights(nw, str(WEIGHTS_DIR / args.save_weights))

if __name__ == "__main__":
    main()