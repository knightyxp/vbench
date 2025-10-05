# python evaluate.py \
#     --dimension 'dynamic_degree' \
#     --videos_path /projects/D2DCRC/xiangpeng/Senorita/obj_removal_videos_vie_top1k/0089_source_869aa598c2736f7f4ae90956e5f50ac1_org.mp4 \
#     --mode=custom_input


# torchrun --nproc_per_node=4 --standalone evaluate.py \
#   --video_json /scratch3/yan204/yxp/InContext-VideoEdit/data/json/obj_swap_top1w.json \
#   --video_base_dir /scratch3/yan204/yxp/Senorita \
#   --dimension 'dynamic_degree'


export CUDA_VISIBLE_DEVICES=1,2

torchrun --nproc_per_node=2 --standalone eval_dynamic_degree_json.py \
  --batch_size 16 \
  --video_json updated_data_obj_grounding_videos.json \
  --video_base_dir /scratch3/yan204/yxp/Senorita \
  --output_path /scratch3/yan204/yxp/vbench/swap_1w_results