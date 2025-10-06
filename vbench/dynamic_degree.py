import argparse
import os
import cv2
import glob
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from easydict import EasyDict as edict
from queue import Queue
from threading import Thread
torch.backends.cudnn.benchmark = True

from vbench.utils import load_dimension_info

from vbench.third_party.RAFT.core.raft import RAFT
from vbench.third_party.RAFT.core.utils_core.utils import InputPadder
import decord
from decord import VideoReader, cpu, gpu
decord.bridge.set_bridge('torch')


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


    # ---------------------
    # Resize helpers
    # ---------------------
    def _get_target_hw(self, h, w):
        # Landscape -> 480x832; Portrait -> 832x480
        if w >= h:
            return 480, 832
        else:
            return 832, 480

    def _resize_batch_nchw(self, frames_nchw):
        """Resize a NCHW batch on CPU to the target size based on orientation.
        frames_nchw: Tensor [N, C, H, W] (float)
        """
        if frames_nchw is None or frames_nchw.numel() == 0:
            return frames_nchw
        h, w = int(frames_nchw.shape[-2]), int(frames_nchw.shape[-1])
        th, tw = self._get_target_hw(h, w)
        if h == th and w == tw:
            return frames_nchw
        return F.interpolate(frames_nchw, size=(th, tw), mode='bilinear', align_corners=False)

    def _resize_single_chw(self, frame_chw):
        """Resize a single CHW tensor to target size keeping type/scale."""
        h, w = int(frame_chw.shape[-2]), int(frame_chw.shape[-1])
        th, tw = self._get_target_hw(h, w)
        if h == th and w == tw:
            return frame_chw
        return F.interpolate(frame_chw[None], size=(th, tw), mode='bilinear', align_corners=False)[0]


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
        # Use decord for accelerated frame reading
        # Prefer GPU decoding if available, else fallback to CPU gracefully
        vr = None
        try:
            local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            vr = VideoReader(video_path, ctx=gpu(local_rank), num_threads=max(4, (os.cpu_count() or 8)//2))
        except Exception:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=max(4, (os.cpu_count() or 8)//2))
        try:
            fps = float(vr.get_avg_fps())
        except Exception:
            fps = 24.0
        interval = max(1, round(fps / 8))
        indices = list(range(0, len(vr), interval))
        if len(indices) == 0:
            return []
        frames = vr.get_batch(indices)
        # Normalize to torch tensor (supports either torch tensor or NDArray)
        if hasattr(frames, 'permute'):
            frames = frames.permute(0, 3, 1, 2).float()
        else:
            frames = torch.from_numpy(frames.asnumpy()).permute(0, 3, 1, 2).float()
        # Resize whole batch based on orientation of original frames
        frames = self._resize_batch_nchw(frames)
        # Convert to list of 1xC,H,W on device
        frame_list = [frames[i].unsqueeze(0).to(self.device) for i in range(frames.shape[0])]
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
            frame = self._resize_single_chw(frame)
            frame = frame[None].to(self.device)
            frame_list.append(frame)
        assert frame_list != []
        return frame_list

    def _iter_pairs_decord_prefetch(self, video_path, batch_size=1, sample_fps_div=8, buffer_size=4):
        """Decode frames with decord, sample by fps/div, and yield adjacent frame pairs in batches.
        Producer decodes and pads on CPU, consumer transfers to GPU and runs RAFT.
        Yields (im1b, im2b) CPU tensors in NCHW already padded.
        """
        # Ensure bridge returns torch when possible
        try:
            decord.bridge.set_bridge('torch')
        except Exception:
            pass
        # Prefer GPU decoding if available, else fallback to CPU gracefully
        vr = None
        try:
            local_rank = int(os.environ.get('LOCAL_RANK', '0'))
            vr = VideoReader(video_path, ctx=gpu(local_rank), num_threads=max(4, (os.cpu_count() or 8)//2))
        except Exception:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=max(4, (os.cpu_count() or 8)//2))
        vlen = len(vr)
        try:
            fps = float(vr.get_avg_fps())
        except Exception:
            fps = 24.0
        interval = max(1, round(fps / sample_fps_div))
        frame_indices = list(range(0, vlen, interval))
        if len(frame_indices) < 2:
            return
        pair_indices = [(frame_indices[i], frame_indices[i+1]) for i in range(len(frame_indices)-1)]

        q: Queue = Queue(maxsize=buffer_size)

        def producer():
            try:
                for start in range(0, len(pair_indices), batch_size):
                    batch_pairs = pair_indices[start:start+batch_size]
                    flat_idx = []
                    for i, j in batch_pairs:
                        flat_idx.append(i)
                        flat_idx.append(j)
                    frames = vr.get_batch(flat_idx)  # (2B, H, W, C)
                    if hasattr(frames, 'permute'):
                        frames = frames.permute(0, 3, 1, 2).float()
                    else:
                        frames = torch.from_numpy(frames.asnumpy()).permute(0, 3, 1, 2).float()
                    # Resize batch to target orientation-based size
                    frames = self._resize_batch_nchw(frames)
                    # Pad with a single padder based on first frame shape
                    padder = InputPadder(frames[0].shape)
                    im1_list, im2_list = [], []
                    for bi in range(0, frames.shape[0], 2):
                        im1, im2 = frames[bi], frames[bi+1]
                        im1p, im2p = padder.pad(im1[None], im2[None])
                        im1_list.append(im1p)
                        im2_list.append(im2p)
                    im1b = torch.cat(im1_list, dim=0)
                    im2b = torch.cat(im2_list, dim=0)
                    q.put((im1b, im2b))
            finally:
                q.put(None)

        th = Thread(target=producer, daemon=True)
        th.start()

        while True:
            item = q.get()
            if item is None:
                break
            yield item

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
        # Use decord async prefetch pipeline with fixed batch size
        batch_size = max(1, int(getattr(dynamic.args, 'batch_size', 1)))
        iters = int(getattr(dynamic.args, 'iters', 20))
        rad_list = []
        with torch.no_grad():
            for im1b_cpu, im2b_cpu in dynamic._iter_pairs_decord_prefetch(vp, batch_size=batch_size):
                im1b = im1b_cpu.to(dynamic.device, non_blocking=True)
                im2b = im2b_cpu.to(dynamic.device, non_blocking=True)
                _, flow_up = dynamic.model(im1b, im2b, iters=iters, test_mode=True)
                for b in range(im1b.shape[0]):
                    rad_list.append(dynamic.get_score(im1b[b:b+1], flow_up[b:b+1]))
        s = float(np.mean(rad_list)) if len(rad_list) > 0 else 0.0
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
        'mixed_precision': True,
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