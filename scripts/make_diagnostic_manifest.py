"""Create a balanced diagnostic subset from an existing split, without touching the source."""
import argparse
import json
import random
from collections import defaultdict, Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='results/splits.json')
    p.add_argument('--output', default='results/diagnostic_splits.json')
    p.add_argument('--train-per-class', type=int, default=600)
    p.add_argument('--val-per-class', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    source = Path(args.source)
    out = Path(args.output)
    if source.resolve() == out.resolve():
        raise ValueError('Output must differ from source manifest')
    data = json.loads(source.read_text(encoding='utf-8'))
    groups = defaultdict(list)
    for row in data['rows']:
        if row['subset'] in ('train', 'val'):
            groups[(row['subset'], int(row['label']))].append(row)
    selected = []
    rng = random.Random(args.seed)
    for subset in ('train', 'val'):
        for label in range(len(data['classes'])):
            count = args.train_per_class if subset == 'train' else args.val_per_class
            rows = groups[(subset, label)]
            if len(rows) < count:
                raise ValueError(f'Only {len(rows)} rows for {subset} class {label}; requested {count}')
            rng.shuffle(rows)
            selected.extend(rows[:count])
    data['rows'] = selected
    data['diagnostic_only'] = True
    data['diagnostic_source_manifest'] = str(source)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f'{out} exists. Refusing to overwrite previous diagnostic data.')
    out.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(json.dumps({'manifest': str(out), 'sizes': dict(Counter(x['subset'] for x in selected)),
                      'diagnostic_only': True}, indent=2))


if __name__ == '__main__':
    main()
