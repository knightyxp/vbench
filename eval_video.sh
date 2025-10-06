# python evaluate.py \
#     --dimension 'dynamic_degree' \
#     --videos_path /projects/D2DCRC/xiangpeng/Senorita/obj_removal_videos_vie_top1k/0089_source_869aa598c2736f7f4ae90956e5f50ac1_org.mp4 \
#     --mode=custom_input


# torchrun --nproc_per_node=4 --standalone evaluate.py \
#   --video_json /scratch3/yan204/yxp/InContext-VideoEdit/data/json/obj_swap_top1w.json \
#   --video_base_dir /scratch3/yan204/yxp/Senorita \
#   --dimension 'dynamic_degree'


export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 --standalone eval_dynamic_degree_json.py \
  --video_json /scratch3/yan204/yxp/Filter_Video_In_context_data/dover_prediction/dover_score_top10w_pairs_grounding_fixed.json \
  --video_base_dir /scratch3/yan204/yxp/Senorita \
  --output_path /scratch3/yan204/yxp/vbench/grounding_10w_results\
  --batch_size 16 \
  --run_id grounding_10w \
  --save_every 100