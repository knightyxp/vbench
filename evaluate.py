import torch
import os
from vbench import VBench
from vbench.distributed import dist_init, print0
from datetime import datetime
import argparse
import json

def parse_args():

    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='VBench', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--output_path",
        type=str,
        default='./evaluation_results/',
        help="output path to save the evaluation results",
    )
    parser.add_argument(
        "--full_json_dir",
        type=str,
        default=f'{CUR_DIR}/vbench/VBench_full_info.json',
        help="path to save the json file that contains the prompt and dimension information",
    )
    parser.add_argument(
        "--videos_path",
        type=str,
        required=False,
        help="folder that contains the sampled videos",
    )
    parser.add_argument(
        "--video_json",
        type=str,
        required=False,
        help="JSON file containing source video paths (list or objects with 'source'/'path'/'video_path')",
    )
    parser.add_argument(
        "--video_base_dir",
        type=str,
        required=False,
        help="Base directory to resolve relative paths in --video_json",
    )
    parser.add_argument(
        "--dimension",
        nargs='+',
        required=True,
        help="list of evaluation dimensions, usage: --dimension <dim_1> <dim_2>",
    )
    parser.add_argument(
        "--load_ckpt_from_local",
        type=bool,
        required=False,
        help="whether load checkpoints from local default paths (assuming you have downloaded the checkpoints locally",
    )
    parser.add_argument(
        "--read_frame",
        type=bool,
        required=False,
        help="whether directly read frames, or directly read videos",
    )
    parser.add_argument(
        "--mode",
        choices=['custom_input', 'vbench_standard', 'vbench_category'],
        default='vbench_standard',
        help="""This flags determine the mode of evaluations, choose one of the following:
        1. "custom_input": receive input prompt from either --prompt/--prompt_file flags or the filename
        2. "vbench_standard": evaluate on standard prompt suite of VBench
        3. "vbench_category": evaluate on specific category
        """,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="None",
        help="""Specify the input prompt
        If not specified, filenames will be used as input prompts
        * Mutually exclusive to --prompt_file.
        ** This option must be used with --mode=custom_input flag
        """
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=False,
        help="""Specify the path of the file that contains prompt lists
        If not specified, filenames will be used as input prompts
        * Mutually exclusive to --prompt.
        ** This option must be used with --mode=custom_input flag
        """
    )
    parser.add_argument(
        "--category",
        type=str,
        required=False,
        help="""This is for mode=='vbench_category'
        The category to evaluate on, usage: --category=animal.
        """,
    )

    ## for dimension specific params ###
    parser.add_argument(
        "--imaging_quality_preprocessing_mode",
        type=str,
        required=False,
        default='longer',
        help="""This is for setting preprocessing in imaging_quality
        1. 'shorter': if the shorter side is more than 512, the image is resized so that the shorter side is 512.
        2. 'longer': if the longer side is more than 512, the image is resized so that the longer side is 512.
        3. 'shorter_centercrop': if the shorter side is more than 512, the image is resized so that the shorter side is 512. 
        Then the center 512 x 512 after resized is used for evaluation.
        4. 'None': no preprocessing
        """,
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    dist_init()
    print0(f'args: {args}')
    device = torch.device("cuda")
    my_VBench = VBench(device, args.full_json_dir, args.output_path)
    
    print0(f'start evaluation')

    current_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

    kwargs = {}

    prompt = []

    if (args.prompt_file is not None) and (args.prompt != "None"):
        raise Exception("--prompt_file and --prompt cannot be used together")
    if (args.prompt_file is not None or args.prompt != "None") and (args.mode!='custom_input'):
        raise Exception("must set --mode=custom_input for using external prompt")

    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompt = json.load(f)
        assert type(prompt) == dict, "Invalid prompt file format. The correct format is {\"video_path\": prompt, ... }"
    elif args.prompt != "None":
        prompt = [args.prompt]

    if args.category != "":
        kwargs['category'] = args.category

    kwargs['imaging_quality_preprocessing_mode'] = args.imaging_quality_preprocessing_mode

    # Validate inputs: require either videos_path or video_json
    if (args.videos_path is None or args.videos_path == "") and (args.video_json is None or args.video_json == ""):
        raise Exception("Either --videos_path or --video_json must be provided")

    # If --video_json is provided, resolve paths and create a temp folder of symlinks
    videos_path_for_eval = args.videos_path
    mode_for_eval = args.mode
    if args.video_json is not None and args.video_json != "":
        json_dir = os.path.dirname(os.path.abspath(args.video_json))
        base_dir = args.video_base_dir if args.video_base_dir else json_dir
        with open(args.video_json, 'r') as f:
            try:
                video_spec = json.load(f)
            except Exception as e:
                raise Exception(f"Failed to load --video_json: {e}")

        # Extract list of raw paths from JSON supporting common formats
        raw_paths = []
        if isinstance(video_spec, list):
            for item in video_spec:
                if isinstance(item, str):
                    raw_paths.append(item)
                elif isinstance(item, dict):
                    for key in ['original_video', 'source_video_path', 'target_video_path', 'source', 'src', 'video', 'video_path', 'path']:
                        if key in item and isinstance(item[key], str):
                            raw_paths.append(item[key])
                            break
        elif isinstance(video_spec, dict):
            # Dict mapping -> take values if they are strings or objects
            for val in video_spec.values():
                if isinstance(val, str):
                    raw_paths.append(val)
                elif isinstance(val, dict):
                    for key in ['original_video', 'source_video_path', 'target_video_path', 'source', 'src', 'video', 'video_path', 'path']:
                        if key in val and isinstance(val[key], str):
                            raw_paths.append(val[key])
                            break
        else:
            raise Exception("Unsupported --video_json format. Use list of strings or list/dict of objects with a path field.")

        # Resolve to absolute local filesystem paths
        abs_paths = []
        for p in raw_paths:
            if isinstance(p, str) and p.strip() != "":
                if p.startswith('http://') or p.startswith('https://'):
                    print0(f"Skip non-local URL path in --video_json: {p}")
                    continue
                candidate = p if os.path.isabs(p) else os.path.join(base_dir, p)
                candidate = os.path.abspath(candidate)
                if os.path.exists(candidate):
                    abs_paths.append(candidate)
                else:
                    print0(f"WARNING: Video path not found, skipping: {candidate}")

        if len(abs_paths) == 0:
            raise Exception("No valid local video paths found from --video_json after resolution")

        # Create a temp input dir within output_path and symlink all videos
        temp_input_dir = os.path.join(args.output_path, f"json_inputs_{current_time}")
        os.makedirs(temp_input_dir, exist_ok=True)
        for idx, vp in enumerate(abs_paths):
            link_name = f"{idx:05d}_" + os.path.basename(vp)
            link_path = os.path.join(temp_input_dir, link_name)
            try:
                if os.path.lexists(link_path):
                    os.unlink(link_path)
                os.symlink(vp, link_path)
            except Exception as e:
                raise Exception(f"Failed to create symlink for {vp} -> {link_path}: {e}")

        videos_path_for_eval = temp_input_dir
        mode_for_eval = 'custom_input'

    my_VBench.evaluate(
        videos_path = videos_path_for_eval,
        name = f'results_{current_time}',
        prompt_list=prompt, # pass in [] to read prompt from filename
        dimension_list = args.dimension,
        local=args.load_ckpt_from_local,
        read_frame=args.read_frame,
        mode=mode_for_eval,
        **kwargs
    )
    print0('done')


if __name__ == "__main__":
    main()
