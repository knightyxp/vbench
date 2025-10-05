import os
import json
import argparse
from datetime import datetime

import torch

from vbench.distributed import dist_init, get_world_size, get_rank, barrier
from vbench.utils import init_submodules, save_json
from vbench.dynamic_degree import DynamicDegree, dynamic_degree


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate dynamic_degree from JSON without copying files')
    parser.add_argument(
        '--video_json', type=str, required=True,
        help='JSON file: list/dict with paths; supports original_video/source_video_path/target_video_path/...')
    parser.add_argument(
        '--video_base_dir', type=str, default=None,
        help='Base dir to resolve relative paths in JSON (defaults to JSON dir)')
    parser.add_argument(
        '--output_path', type=str, default='./evaluation_results/',
        help='Directory to save results JSON')
    parser.add_argument(
        '--local', action='store_true',
        help='Use local checkpoints (download if missing)')
    return parser.parse_args()


def load_paths_from_json(json_path: str, base_dir: str | None):
    json_dir = os.path.dirname(os.path.abspath(json_path))
    base = base_dir if base_dir else json_dir
    with open(json_path, 'r') as f:
        spec = json.load(f)

    candidates = []
    def extract_from_obj(obj):
        for key in ['original_video', 'source_video_path', 'target_video_path', 'source', 'src', 'video', 'video_path', 'path']:
            if isinstance(obj, dict) and key in obj and isinstance(obj[key], str):
                return obj[key]
        return None

    if isinstance(spec, list):
        for item in spec:
            if isinstance(item, str):
                candidates.append(item)
            elif isinstance(item, dict):
                p = extract_from_obj(item)
                if p: candidates.append(p)
    elif isinstance(spec, dict):
        for val in spec.values():
            if isinstance(val, str):
                candidates.append(val)
            elif isinstance(val, dict):
                p = extract_from_obj(val)
                if p: candidates.append(p)
    else:
        raise Exception('Unsupported JSON format')

    abs_paths = []
    for p in candidates:
        if not isinstance(p, str) or p.strip() == '':
            continue
        if p.startswith('http://') or p.startswith('https://'):
            continue
        cand = p if os.path.isabs(p) else os.path.join(base, p)
        cand = os.path.abspath(cand)
        if os.path.exists(cand):
            abs_paths.append(cand)
    if len(abs_paths) == 0:
        raise Exception('No valid local video paths found from JSON')
    return abs_paths


def main():
    args = parse_args()
    dist_init()

    device = torch.device('cuda')
    # Prepare model
    submods = init_submodules(['dynamic_degree'], local=args.local, read_frame=False)
    model_path = submods['dynamic_degree']['model']
    dd_args = type('cfg', (), {
        'model': model_path,
        'small': False,
        'mixed_precision': False,
        'alternate_corr': False,
    })
    dynamic = DynamicDegree(dd_args, device)

    # Resolve paths and distribute
    all_paths = load_paths_from_json(args.video_json, args.video_base_dir)
    world = get_world_size()
    rank = get_rank()
    shard = all_paths[rank::world]

    # Compute local results
    _, local_results = dynamic_degree(dynamic, shard)

    # Simple gather via file writes per rank; coordinator will merge
    os.makedirs(args.output_path, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    part_path = os.path.join(args.output_path, f'dynamic_degree_parts_{ts}_rank{rank}.json')
    save_json(local_results, part_path)

    # Synchronize before merge to ensure all part files are written
    barrier()
    # Coordinator merges
    if rank == 0:
        # Wait for others via simple barrier in distributed module handled by torchrun termination
        # Merge parts present
        merged = []
        for r in range(world):
            rp = os.path.join(args.output_path, f'dynamic_degree_parts_{ts}_rank{r}.json')
            if os.path.exists(rp):
                with open(rp, 'r') as f:
                    merged.extend(json.load(f))
        avg = float(sum(d['dynamic_score'] for d in merged) / len(merged))
        out = {
            'dimension': 'dynamic_degree',
            'average_score': avg,
            'results': merged,
            'total_videos': len(merged),
            'timestamp': ts,
            'source_json': os.path.abspath(args.video_json),
        }
        save_json(out, os.path.join(args.output_path, f'dynamic_degree_eval_{ts}.json'))


if __name__ == '__main__':
    main()


