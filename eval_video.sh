# python evaluate.py \
#     --dimension 'dynamic_degree' \
#     --videos_path /projects/D2DCRC/xiangpeng/Senorita/obj_removal_videos_vie_top1k/0089_source_869aa598c2736f7f4ae90956e5f50ac1_org.mp4 \
#     --mode=custom_input

export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 --standalone evaluate.py \
    --dimension 'dynamic_degree' \
    --videos_path /projects/D2DCRC/xiangpeng/Senorita/obj_removal_videos_vie_top1k \
    --mode=custom_input