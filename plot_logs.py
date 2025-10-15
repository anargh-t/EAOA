import re
import os
import argparse
from typing import List, Tuple
from glob import glob

import matplotlib.pyplot as plt


def parse_log(filepath: str) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """Parse a training log and extract metrics per query round.

    Returns lists aligned by query round where available:
      - acc_id:  C-way accuracy (%)
      - acc_ood: (C+1)-way accuracy (%)
      - prec:    query precision
      - rec:     query recall
      - k1_values: adapted k1 values
    """
    acc_id, acc_ood, prec, rec, k1_values = [], [], [], [], []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if 'Model_ID | Accuracy (%)' in line:
                m = re.search(r'Accuracy \(%\): ([0-9.]+)', line)
                if m:
                    acc_id.append(float(m.group(1)))
            elif 'Model_ID_w_OOD | Accuracy (%)' in line:
                m = re.search(r'Accuracy \(%\): ([0-9.]+)', line)
                if m:
                    acc_ood.append(float(m.group(1)))
            elif 'Query Strategy:' in line and 'Query Precision:' in line:
                mp = re.search(r'Query Precision: ([0-9.]+)', line)
                mr = re.search(r'Query Recall: ([0-9.]+)', line)
                if mp:
                    prec.append(float(mp.group(1)))
                if mr:
                    rec.append(float(mr.group(1)))
            elif 'Current k_t value is' in line:
                try:
                    k1_values.append(float(line.strip().split()[-1]))
                except Exception:
                    pass
    return acc_id, acc_ood, prec, rec, k1_values


def plot_curves(logfile: str, outdir: str) -> None:
    """Single-run plots.

    Generates accuracy_vs_round.png using C-way accuracy.
    """
    os.makedirs(outdir, exist_ok=True)
    acc_id, acc_ood, prec, rec, k1_values = parse_log(logfile)

    # Accuracy vs query round (C-way only)
    plt.figure(figsize=(6.5, 4.5))
    if acc_id:
        plt.plot(range(1, len(acc_id) + 1), acc_id, label='Acc ID (C-way)')
    plt.xlabel('Query round')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Query Round')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'accuracy_vs_round.png'), dpi=200)
    plt.close()


def plot_accuracy_comparison(logfiles: List[str], outdir: str, labels: List[str] = None) -> None:
    """Comparison plot across logs using C-way accuracy.

    Saves acc_vs_cycles_C.png.
    """
    """Plot test accuracy vs AL cycles for multiple runs on one figure.

    Args:
        logfiles: list of log paths
        outdir: output dir
        use_ood: if True, use (C+1)-way accuracy; else use C-way accuracy
        labels: optional series labels for legend (must match len(logfiles))
    """
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    # Helper to shorten labels by removing common prefix/suffix
    def _strip_common(parts: List[str]) -> List[str]:
        if not parts:
            return parts
        # Common prefix
        prefix = os.path.commonprefix(parts)
        # Common suffix via reversed strings
        rev = [p[::-1] for p in parts]
        suffix_rev = os.path.commonprefix(rev)
        suffix = suffix_rev[::-1]
        stripped = []
        for p in parts:
            s = p
            if prefix:
                s = s[len(prefix):]
            if suffix and len(suffix) < len(s):
                s = s[:-len(suffix)]
            # Fallback if empty
            stripped.append(s if s else p)
        return stripped

    basenames = [os.path.basename(p) for p in logfiles]
    shortnames = _strip_common(basenames)

    plotted_any = False
    plotted_count = 0
    for i, lf in enumerate(logfiles):
        if not os.path.exists(lf):
            print(f'[WARN] File not found, skipping: {lf}')
            continue
        acc_id, acc_ood, prec, rec, k1_values = parse_log(lf)
        y = acc_id
        if not y:
            print(f'[WARN] No accuracy entries parsed from: {lf}')
            continue
        x = list(range(1, len(y) + 1))
        default_label = shortnames[i] if i < len(shortnames) else os.path.basename(lf)
        label = labels[i] if labels and i < len(labels) else default_label
        plt.plot(x, y, marker='o', linewidth=1.5, label=label)
        plotted_any = True
        plotted_count += 1
    if not plotted_any:
        print('[ERROR] No valid log files to plot. Provide existing .log paths or patterns.')
        return
    plt.xlabel('AL cycle (query round)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs AL Cycles (C-way)')
    plt.grid(True, alpha=0.3)
    # Legend: only show if comparing 2+ series; place outside to avoid overlap
    if plotted_count >= 2:
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    plt.tight_layout()
    fname = 'acc_vs_cycles_C.png'
    plt.savefig(os.path.join(outdir, fname), dpi=200)
    plt.close()


def plot_precision_vs_final_acc(logfiles: List[str], outdir: str, labels: List[str] = None) -> None:
    """Scatter: mean query precision (x) vs final test accuracy (y) across logs.

    Args:
        logfiles: list of log paths
        outdir: output dir
        labels: optional labels for annotations/legend
    """
    os.makedirs(outdir, exist_ok=True)

    # Helper to shorten labels by removing common prefix/suffix
    def _strip_common(parts: List[str]) -> List[str]:
        if not parts:
            return parts
        prefix = os.path.commonprefix(parts)
        rev = [p[::-1] for p in parts]
        suffix_rev = os.path.commonprefix(rev)
        suffix = suffix_rev[::-1]
        stripped = []
        for p in parts:
            s = p
            if prefix:
                s = s[len(prefix):]
            if suffix and len(suffix) < len(s):
                s = s[:-len(suffix)]
            stripped.append(s if s else p)
        return stripped

    xs, ys, names = [], [], []
    basenames = [os.path.basename(p) for p in logfiles]
    shortnames = _strip_common(basenames)
    for i, lf in enumerate(logfiles):
        if not os.path.exists(lf):
            print(f'[WARN] File not found, skipping: {lf}')
            continue
        acc_id, acc_ood, prec, rec, _ = parse_log(lf)
        if not prec:
            print(f'[WARN] No precision entries in: {lf}')
            continue
        y_series = acc_id
        if not y_series:
            print(f'[WARN] No accuracy entries in: {lf}')
            continue
        mean_p = sum(prec) / len(prec)
        final_acc = y_series[-1]
        xs.append(mean_p)
        ys.append(final_acc)
        default_label = shortnames[i] if i < len(shortnames) else os.path.basename(lf)
        names.append(labels[i] if labels and i < len(labels) else default_label)

    if not xs:
        print('[ERROR] No valid points for precision vs final accuracy plot.')
        return

    plt.figure(figsize=(6.5, 4.5))
    plt.scatter(xs, ys, c='tab:blue', edgecolors='k')
    for x, y, n in zip(xs, ys, names):
        plt.annotate(n, (x, y), textcoords='offset points', xytext=(6, 6), fontsize=8, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.7', alpha=0.8))
    plt.xlabel('Mean Query Precision (across AL rounds)')
    plt.ylabel('Final Test Accuracy (%)')
    plt.title('Mean Query Precision vs Final Test Accuracy (C-way)')
    # Axis limits: tighten around data; special handling for single point
    if len(xs) == 1:
        xpad = 0.02
        ypad = 1.0
        plt.xlim(xs[0] - xpad, xs[0] + xpad)
        plt.ylim(ys[0] - ypad, ys[0] + ypad)
    else:
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        xpad = max((xmax - xmin) * 0.08, 0.01)
        ypad = max((ymax - ymin) * 0.08, 0.5)
        plt.xlim(xmin - xpad, xmax + xpad)
        plt.ylim(ymin - ypad, ymax + ypad)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = 'precision_vs_final_acc_C.png'
    plt.savefig(os.path.join(outdir, fname), dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Plots: acc_vs_cycles_C.png, precision_vs_final_acc_C.png, accuracy_vs_round.png')
    parser.add_argument('logfile', type=str, nargs='+', help='Path(s)/patterns/dirs to run log file(s) (*.log)')
    parser.add_argument('--outdir', type=str, default='plots', help='Directory to save plots')
    parser.add_argument('--compare', action='store_true', help='Plot comparison across multiple logs (accuracy vs AL cycles, C-way)')
    parser.add_argument('--labels', type=str, nargs='*', help='Optional labels for comparison legend (same order as logfiles)')
    parser.add_argument('--dataset-filter', type=str, help='Keep only logs whose filename contains this token (e.g., cifar10)')
    parser.add_argument('--scatter-prec-vs-acc', action='store_true', help='Scatter of mean query precision vs final test accuracy (C-way)')
    args = parser.parse_args()

    # Resolve inputs: expand directories and glob patterns into .log files
    resolved: List[str] = []
    for entry in args.logfile:
        if os.path.isdir(entry):
            resolved.extend(sorted(glob(os.path.join(entry, '*.log'))))
        else:
            expanded = glob(entry)
            if expanded:
                resolved.extend(sorted(expanded))
            else:
                resolved.append(entry)

    # Optional dataset filter by filename token
    if args.dataset_filter:
        token = args.dataset_filter.lower()
        resolved = [p for p in resolved if token in os.path.basename(p).lower()]
        if not resolved:
            print(f"[ERROR] No log files matched dataset-filter='{token}'.")
            return

    if args.compare and len(resolved) >= 2:
        plot_accuracy_comparison(resolved, args.outdir, labels=args.labels)
    elif args.scatter_prec_vs_acc and len(resolved) >= 1:
        plot_precision_vs_final_acc(resolved, args.outdir, labels=args.labels)
    else:
        # default: process the first resolved logfile for single-run plots
        if not resolved:
            print('[ERROR] No log files found. Provide at least one .log file.')
            return
        if not os.path.exists(resolved[0]):
            print(f"[ERROR] File not found: {resolved[0]}")
            return
        plot_curves(resolved[0], args.outdir)
    print(f'Plots saved to: {os.path.abspath(args.outdir)}')


if __name__ == '__main__':
    main()


