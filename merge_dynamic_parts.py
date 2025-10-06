#!/usr/bin/env python3
import os
import re
import json
import glob
import argparse
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Merge dynamic_degree per-rank parts into one JSON')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing part files')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to write merged JSON (default=input_dir)')
    parser.add_argument('--run_id', type=str, default='', help='Run ID to select files (e.g., grounding_10w). If empty, auto-detect latest group')
    parser.add_argument('--ranks', type=str, default='0,1,2,3', help='Comma-separated rank ids to merge (used when run_id not provided)')
    parser.add_argument('--outfile', type=str, default='', help='Output filename (optional). Defaults to dynamic_degree_eval_{run_id or timestamp}.json')
    return parser.parse_args()


def detect_groups(input_dir: str):
    pattern = os.path.join(input_dir, 'dynamic_degree_parts_*_rank*.json*')
    files = glob.glob(pattern)
    groups = {}
    rgx = re.compile(r'dynamic_degree_parts_(.+)_rank\d+\.(jsonl|json)$')
    for p in files:
        base = os.path.basename(p)
        m = rgx.match(base)
        if not m:
            continue
        gid = m.group(1)
        groups.setdefault(gid, []).append(p)
    return groups


def pick_latest_per_rank(input_dir: str, ranks: list[int]):
    selected = []
    for r in ranks:
        candidates = []
        pattern_jsonl = os.path.join(input_dir, f'dynamic_degree_parts_*_rank{r}.jsonl')
        pattern_json = os.path.join(input_dir, f'dynamic_degree_parts_*_rank{r}.json')
        candidates.extend(glob.glob(pattern_jsonl))
        candidates.extend(glob.glob(pattern_json))
        if not candidates:
            continue
        best = max(candidates, key=lambda p: os.path.getmtime(p))
        selected.append(best)
    return sorted(selected)


def pick_group(groups: dict[str, list[str]]):
    if not groups:
        return None, []
    # Prefer group with most files; break ties by latest mtime
    best_gid = None
    best_files = []
    best_key = (-1, -1.0)
    for gid, paths in groups.items():
        count = len(paths)
        latest_mtime = max(os.path.getmtime(p) for p in paths)
        key = (count, latest_mtime)
        if key > best_key:
            best_key = key
            best_gid = gid
            best_files = paths
    return best_gid, sorted(best_files)


def read_part_file(path: str):
    records = []
    try:
        if path.endswith('.jsonl'):
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if isinstance(rec, dict):
                            records.append(rec)
                    except Exception:
                        pass
        else:
            with open(path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for rec in data:
                        if isinstance(rec, dict):
                            records.append(rec)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
    return records


def main():
    args = parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else input_dir
    os.makedirs(output_dir, exist_ok=True)

    files = []
    group_id = ''
    if args.run_id:
        group_id = args.run_id
        # Prefer JSONL, fallback to JSON
        files = sorted(glob.glob(os.path.join(input_dir, f'dynamic_degree_parts_{group_id}_rank*.jsonl')))
        if not files:
            files = sorted(glob.glob(os.path.join(input_dir, f'dynamic_degree_parts_{group_id}_rank*.json')))
    else:
        # Merge by latest file per specified rank, ignoring timestamp in filename
        try:
            ranks = [int(x) for x in args.ranks.split(',') if x.strip()!='']
        except Exception:
            ranks = [0,1,2,3]
        files = pick_latest_per_rank(input_dir, ranks)

    if not files:
        raise SystemExit(f"No part files found in {input_dir} (run_id='{group_id}' ranks='{getattr(args,'ranks','')}' )")

    print(f"Merging {len(files)} part files for group '{group_id}':")
    for p in files:
        print(f" - {os.path.basename(p)}")

    merged = []
    seen = set()
    for p in files:
        for rec in read_part_file(p):
            if not isinstance(rec, dict):
                continue
            vp = rec.get('video_path')
            if not vp or vp in seen:
                continue
            score_val = rec.get('dynamic_score', 0.0)
            try:
                score = float(score_val)
            except Exception:
                score = 0.0
            merged.append({'video_path': vp, 'dynamic_score': score})
            seen.add(vp)

    total = len(merged)
    avg = float(sum(r['dynamic_score'] for r in merged) / total) if total > 0 else 0.0

    out_name = args.outfile.strip() if args.outfile else 'dynamic_degree_eval_merged.json'
    out_path = os.path.join(output_dir, out_name)
    out_obj = {
        'dimension': 'dynamic_degree',
        'average_score': avg,
        'results': merged,
        'total_videos': total,
        'run_id': group_id,
        'source_parts_dir': input_dir,
    }
    with open(out_path, 'w') as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)
    print(f"Wrote merged results -> {out_path} (total_videos={total}, average_score={avg:.6f})")


if __name__ == '__main__':
    main()


