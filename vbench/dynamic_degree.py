import argparse
import os
import cv2
import glob
import numpy as np
import torch
from tqdm import tqdm
from easydict import EasyDict as edict
from queue import Queue
from threading import Thread

from vbench.utils import load_dimension_info

from vbench.third_party.RAFT.core.raft import RAFT
from vbench.third_party.RAFT.core.utils_core.utils import InputPadder


from .distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)


class DynamicDegree:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.load_model()
    

    def load_model(self):
        self.model = RAFT(self.args)
        ckpt = torch.load(self.args.model, map_location="cpu")
        new_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        self.model.load_state_dict(new_ckpt)
        self.model.to(self.device)
        self.model.eval()


    def get_score(self, img, flo):
        img = img[0].permute(1,2,0).cpu().numpy()
        flo = flo[0].permute(1,2,0).cpu().numpy()

        u = flo[:,:,0]
        v = flo[:,:,1]
        rad = np.sqrt(np.square(u) + np.square(v))
        
        h, w = rad.shape
        rad_flat = rad.flatten()
        cut_index = int(h*w*0.05)

        max_rad = np.mean(abs(np.sort(-rad_flat))[:cut_index])

        return max_rad.item()


    def set_params(self, frame, count):
        scale = min(list(frame.shape)[-2:])
        self.params = {"thres":6.0*(scale/256.0), "count_num":round(4*(count/16.0))}


    def infer(self, video_path):
        # with torch.no_grad():
        #     if video_path.endswith('.mp4'):
        #         frames = self.get_frames(video_path)
        #     elif os.path.isdir(video_path):
        #         frames = self.get_frames_from_img_folder(video_path)
        #     else:
        #         raise NotImplementedError
        #     self.set_params(frame=frames[0], count=len(frames))
        #     static_score = []
        #     for image1, image2 in zip(frames[:-1], frames[1:]):
        #         padder = InputPadder(image1.shape)
        #         image1, image2 = padder.pad(image1, image2)
        #         _, flow_up = self.model(image1, image2, iters=20, test_mode=True)
        #         max_rad = self.get_score(image1, flow_up)
        #         static_score.append(max_rad)
        #     whether_move = self.check_move(static_score)
        #     return whether_move
        # 1. 读取并采样帧
        frames = (self.get_frames(video_path)
                  if video_path.endswith('.mp4')
                  else self.get_frames_from_img_folder(video_path))
        # 2. 计算每对相邻帧的最大光流幅值（固定 batch）
        rad_list = []
        with torch.no_grad():
            total_pairs = len(frames) - 1
            if total_pairs <= 0:
                return 0.0
            batch_size = max(1, int(getattr(self.args, 'batch_size', 1)))
            iters = int(getattr(self.args, 'iters', 20))
            start_idx = 0
            while start_idx < total_pairs:
                cur_bs = min(batch_size, total_pairs - start_idx)
                indices = list(range(start_idx, start_idx + cur_bs))
                # 使用第一对的尺寸构建一次 padder
                ref_im1 = frames[indices[0]]
                padder = InputPadder(ref_im1.shape)
                im1_list = []
                im2_list = []
                for j in indices:
                    im1, im2 = frames[j], frames[j+1]
                    im1p, im2p = padder.pad(im1, im2)
                    im1_list.append(im1p)
                    im2_list.append(im2p)
                im1b = torch.cat(im1_list, dim=0)
                im2b = torch.cat(im2_list, dim=0)
                _, flow_up = self.model(im1b, im2b, iters=iters, test_mode=True)
                # 累计每个样本的得分
                for b in range(im1b.shape[0]):
                    rad_list.append(self.get_score(im1b[b:b+1], flow_up[b:b+1]))
                start_idx += cur_bs
        # 3. 返回平均幅值
        return float(np.mean(rad_list))

    def check_move(self, score_list):
        thres = self.params["thres"]
        count_num = self.params["count_num"]
        count = 0
        for score in score_list:
            if score > thres:
                count += 1
            if count >= count_num:
                return True
        return False


    def get_frames(self, video_path):
        # Deprecated: replaced by decord pipeline for performance
        frame_list = []
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS) # get fps
        interval = max(1, round(fps / 8))
        while video.isOpened():
            success, frame = video.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to rgb
                frame = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
                frame = frame[None].to(self.device)
                frame_list.append(frame)
            else:
                break
        video.release()
        assert frame_list != []
        frame_list = self.extract_frame(frame_list, interval)
        return frame_list 
    
    
    def extract_frame(self, frame_list, interval=1):
        extract = []
        for i in range(0, len(frame_list), interval):
            extract.append(frame_list[i])
        return extract


    def get_frames_from_img_folder(self, img_folder):
        exts = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 
        'tiff', 'JPG', 'PNG', 'JPEG', 'BMP', 
        'TIF', 'TIFF']
        frame_list = []
        imgs = sorted([p for p in glob.glob(os.path.join(img_folder, "*")) if os.path.splitext(p)[1][1:] in exts])
        # imgs = sorted(glob.glob(os.path.join(img_folder, "*.png")))
        for img in imgs:
            frame = cv2.imread(img, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
            frame = frame[None].to(self.device)
            frame_list.append(frame)
        assert frame_list != []
        return frame_list

    # decord pipeline removed per request



def dynamic_degree(dynamic, video_list):
    # sim = []
    # video_results = []
    # for video_path in tqdm(video_list, disable=get_rank() > 0):
    #     score_per_video = dynamic.infer(video_path)
    #     video_results.append({'video_path': video_path, 'video_results': score_per_video})
    #     sim.append(score_per_video)
    # avg_score = np.mean(sim)
    # return avg_score, video_results
    scores = []
    results = []
    for vp in tqdm(video_list, disable=get_rank()>0):
        s = dynamic.infer(vp)
        results.append({'video_path': vp, 'dynamic_score': s})
        scores.append(s)
    return float(np.mean(scores)) if len(scores) > 0 else 0.0, results


def compute_dynamic_degree(json_dir, device, submodules_list, **kwargs):
    # model_path = submodules_list["model"] 
    # # set_args
    # args_new = edict({"model":model_path, "small":False, "mixed_precision":False, "alternate_corr":False})
    # dynamic = DynamicDegree(args_new, device)
    # video_list, _ = load_dimension_info(json_dir, dimension='dynamic_degree', lang='en')
    # video_list = distribute_list_to_rank(video_list)
    # all_results, video_results = dynamic_degree(dynamic, video_list)
    # if get_world_size() > 1:
    #     video_results = gather_list_of_dict(video_results)
    #     all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    # return all_results, video_results

    model_path = submodules_list['model']
    args_new = edict({
        'model': model_path,
        'small': False,
        'mixed_precision': False,
        'alternate_corr': False,
    })
    dynamic = DynamicDegree(args_new, device)
    video_list, _ = load_dimension_info(json_dir, dimension='dynamic_degree', lang='en')
    video_list = distribute_list_to_rank(video_list)

    # 1) 本地计算
    _, local_results = dynamic_degree(dynamic, video_list)
    # 2) 多卡聚合
    if get_world_size() > 1:
        gathered = gather_list_of_dict(local_results)
        if isinstance(gathered, list) and isinstance(gathered[0], list):
            video_results = [item for sub in gathered for item in sub]
        else:
            video_results = gathered
    else:
        video_results = local_results

    # 3) 计算全局平均分
    all_scores = [d['dynamic_score'] for d in video_results]
    all_results = float(np.mean(all_scores))
    return all_results, video_results