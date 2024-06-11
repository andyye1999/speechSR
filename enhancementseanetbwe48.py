'''
Author: yehongcen && as_yhc@163.com
Date: 2024-04-23 15:36:38
LastEditors: redust as_yhc@163.com
LastEditTime: 2024-04-23 16:27:58
FilePath: \lastdance\enhancementseanetbwe48.py
Description: 第三章模型预测程序
Copyright (c) 2024 by yehongcen, All Rights Reserved. 
'''
import argparse
import json
import os

import librosa
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import soundfile as sf

from util.utils import initialize_config, load_checkpoint


if __name__ == "__main__":
    """
    Parameters
    """
    parser = argparse.ArgumentParser("seanet: BWE")
    # 配置文件位置
    parser.add_argument("-C", "--config", default= "D:\\yhc\\BWE\\config\\enhancement\\seanetbwe.json", type=str, help="Model and dataset for enhancement (*.json).")
    # 输出文件位置
    parser.add_argument("-D", "--device", default="0", type=str, help="GPU for speech enhancement. default: CPU")
    # 输出文件位置
    parser.add_argument("-O", "--output_dir", default= "D:\\yhc\\BWE\\enhanced\\show", type=str, help="Where are audio save.")
    # 模型位置
    parser.add_argument("-M", "--model_checkpoint_path", default= "D:\\yhc\\BWE\\cbbrn1bwe12_48\\checkpoints\\best_model.tar", type=str, help="Checkpoint.")
    args = parser.parse_args()

    """
    Preparation
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    config = json.load(open(args.config))
    model_checkpoint_path = args.model_checkpoint_path
    output_dir = args.output_dir
    assert os.path.exists(output_dir), "Enhanced directory should be exist."

    """
    DataLoader
    """
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
    # device = torch.device("cuda:0" : "cuda:0")
    dataloader = DataLoader(dataset=initialize_config(config["dataset"]), batch_size=1, num_workers=1)

    """
    Model
    """
    model = initialize_config(config["model"])
    # model = GeneratorEBEN(bands_nbr=4, pqmf_ks=32)
    model.load_state_dict(load_checkpoint(model_checkpoint_path, device))
    model.to(device)
    model.eval()

    """
    Enhancement
    """
    sample_length = config["custom"]["sample_length"]
    for mixture,  name in tqdm(dataloader):
        assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
        name = name[0]
        padded_length = 0
        mixture = mixture.to(device)  # [1, 1, T]
        # clean = clean.to(device)
        # max_bone = max_bone.to(device)
        # max_air = max_air.to(device)
        # mixture = model.cut_tensor(mixture)
        # clean = model.cut_tensor(clean)
        if mixture.size(-1) % sample_length != 0:
            padded_length = sample_length - (mixture.size(-1) % sample_length)
            mixture = torch.cat([mixture, torch.zeros(1, 1, padded_length, device=device)], dim=-1) # 如果长度不整除，切除多余音频

        assert mixture.size(-1) % sample_length == 0 and mixture.dim() == 3
        mixture_chunks = list(torch.split(mixture, sample_length, dim=-1))

        enhanced_chunks = []
        for chunk in mixture_chunks:
            enhanced_speech = model(chunk)
            # enhanced_speech = enhanced_speech * max_bone
            enhanced_chunks.append(enhanced_speech) # 通过模型
        enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T] # 拼接片段
        enhanced = enhanced if padded_length == 0 else enhanced[:, :, :-padded_length]

        # clean = clean * max_air
        enhanced = enhanced.detach().cpu()
        enhanced = enhanced.reshape(-1).numpy()
        # clean = clean.cpu().numpy().reshape(-1)
        # mixture = mixture.cpu().numpy().reshape(-1)
        output_path = os.path.join(output_dir, f"{name}.wav") # 输出
        # librosa.output.write_wav(output_path, enhanced, sr=16000)
        sf.write(output_path, enhanced, 48000)
