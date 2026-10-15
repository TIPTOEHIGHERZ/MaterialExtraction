import os
import argparse

import numpy as np
import omegaconf


def parse_args():
    parser = argparse.ArgumentParser(
        description='Produce a reproducible logical train/test split for a PBR render dataset. '
                    'Writes a yaml of sub-directory names (no files are moved).'
    )
    parser.add_argument('--data_path', required=True,
                        help='Dataset root; its first-level sub-directories are the split units.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train_ratio', type=float, default=0.9,
                       help='Fraction of sub-directories used for training (default: 0.9). '
                            'Ignored when --num_train is given.')
    group.add_argument('--num_train', type=int, default=None,
                       help='Absolute number of training sub-directories. Overrides --train_ratio.')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed; combined with a sorted file list for cross-machine reproducibility.')
    parser.add_argument('--output', default=None,
                        help='Output yaml path (default: <data_path>/seperate.yaml).')
    return parser.parse_args()


def main():
    args = parse_args()

    files = sorted(
        f for f in os.listdir(args.data_path)
        if os.path.isdir(os.path.join(args.data_path, f))
    )
    num_total = len(files)
    assert num_total > 0, f'no sub-directory found under {args.data_path}'

    if args.num_train is not None:
        num_train = args.num_train
        mode = f'num_train={num_train}'
    else:
        num_train = int(round(num_total * args.train_ratio))
        mode = f'train_ratio={args.train_ratio}'
    num_train = max(0, min(num_train, num_total))

    rng = np.random.default_rng(args.seed)
    train_idx = set(rng.choice(num_total, num_train, replace=False).tolist())

    train_files = [f for i, f in enumerate(files) if i in train_idx]
    test_files = [f for i, f in enumerate(files) if i not in train_idx]

    output = args.output or os.path.join(args.data_path, 'seperate.yaml')
    omegaconf.OmegaConf.save({
        'train': train_files,
        'test': test_files,
        'meta': {
            'seed': args.seed,
            'mode': mode,
            'num_total': num_total,
            'num_train': len(train_files),
            'num_test': len(test_files),
        },
    }, output)
    print(f'{output}: {len(train_files)} train / {len(test_files)} test (seed={args.seed}, {mode})')


if __name__ == '__main__':
    main()
