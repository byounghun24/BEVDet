#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize TensorRT trtexec layer profile JSON.')
    parser.add_argument('profile_json', help='Path to trtexec --exportProfile JSON')
    parser.add_argument(
        '--topk', type=int, default=30, help='Number of top layers to plot')
    parser.add_argument(
        '--out-png',
        default=None,
        help='Output PNG path (default: <profile_json_stem>_topk.png)')
    parser.add_argument(
        '--out-csv',
        default=None,
        help='Output CSV path (default: <profile_json_stem>_summary.csv)')
    return parser.parse_args()


def _try_float(v):
    try:
        return float(v)
    except Exception:
        return None


def _extract_ms(entry):
    candidate_keys = [
        'timeMs', 'averageMs', 'avgMs', 'latencyMs', 'medianMs', 'meanMs',
        'gpuMs', 'computeMs', 'ms'
    ]
    for k in candidate_keys:
        if k in entry:
            x = _try_float(entry[k])
            if x is not None:
                return x
    for k, v in entry.items():
        if 'ms' in k.lower():
            x = _try_float(v)
            if x is not None:
                return x
    return None


def _extract_name(entry):
    for k in ['name', 'layerName', 'layer', 'opName', 'nodeName']:
        if k in entry and entry[k]:
            return str(entry[k])
    return None


def _walk_json(node):
    if isinstance(node, dict):
        name = _extract_name(node)
        ms = _extract_ms(node)
        if name is not None and ms is not None:
            yield name, ms
        for v in node.values():
            yield from _walk_json(v)
    elif isinstance(node, list):
        for x in node:
            yield from _walk_json(x)


def main():
    args = parse_args()
    profile_path = Path(args.profile_json)
    stem = profile_path.with_suffix('')
    out_png = Path(args.out_png) if args.out_png else Path(f'{stem}_topk.png')
    out_csv = Path(args.out_csv) if args.out_csv else Path(f'{stem}_summary.csv')

    with open(profile_path, 'r') as f:
        obj = json.load(f)

    agg = defaultdict(lambda: [0.0, 0])
    for name, ms in _walk_json(obj):
        agg[name][0] += ms
        agg[name][1] += 1

    if not agg:
        raise RuntimeError(
            'No layer timing entries found in JSON. Verify trtexec --exportProfile output.')

    rows = []
    total_ms = 0.0
    for name, (sum_ms, cnt) in agg.items():
        avg_ms = sum_ms / cnt
        rows.append((name, sum_ms, avg_ms, cnt))
        total_ms += sum_ms

    rows.sort(key=lambda x: x[1], reverse=True)

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['layer', 'total_ms', 'avg_ms', 'count', 'share_percent'])
        for name, sum_ms, avg_ms, cnt in rows:
            share = 100.0 * sum_ms / total_ms if total_ms > 0 else 0.0
            writer.writerow([name, f'{sum_ms:.6f}', f'{avg_ms:.6f}', cnt, f'{share:.4f}'])

    topk = rows[:max(1, args.topk)]
    labels = [x[0] for x in topk][::-1]
    vals = [x[1] for x in topk][::-1]

    import matplotlib.pyplot as plt

    fig_h = max(6, int(len(labels) * 0.35))
    fig, ax = plt.subplots(figsize=(16, fig_h))
    bars = ax.barh(labels, vals)
    ax.set_title(f'TensorRT Layer Time Top-{len(labels)}')
    ax.set_xlabel('Total time (ms)')
    ax.set_ylabel('Layer')
    ax.grid(axis='x', alpha=0.25)

    for bar, v in zip(bars, vals):
        ax.text(v, bar.get_y() + bar.get_height() * 0.5, f' {v:.3f} ms', va='center')

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)

    top1_name, top1_sum, _, _ = rows[0]
    print(f'Saved CSV: {out_csv}')
    print(f'Saved PNG: {out_png}')
    print(f'Total profiled layer-time sum: {total_ms:.3f} ms')
    print(f'Top hotspot: {top1_name} ({top1_sum:.3f} ms, {100.0 * top1_sum / total_ms:.2f}%)')


if __name__ == '__main__':
    main()
