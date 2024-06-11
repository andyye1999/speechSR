'''
Author: yehongcen && as_yhc@163.com
Date: 2024-04-23 15:36:38
LastEditors: redust as_yhc@163.com
LastEditTime: 2024-04-23 16:25:57
FilePath: \lastdance\train.py
Description: 训练代码
Copyright (c) 2024 by yehongcen, All Rights Reserved. 
'''
import argparse
import os

import json5
import numpy as np
import torch
from torch.utils.data import DataLoader
from util.utils import initialize_config,collate_fn
import tqdm

'''
description: 主函数
param {*} config
param {*} resume
return {*}
'''
def main(config, resume):
    torch.manual_seed(config["seed"])  # for both CPU and GPU
    np.random.seed(config["seed"])

    train_dataloader = DataLoader(
        dataset=initialize_config(config["train_dataset"]),
        batch_size=config["train_dataloader"]["batch_size"],
        num_workers=config["train_dataloader"]["num_workers"],
        shuffle=config["train_dataloader"]["shuffle"],
        pin_memory=config["train_dataloader"]["pin_memory"]
        # ,collate_fn=collate_fn
    )

    valid_dataloader = DataLoader(
        dataset=initialize_config(config["validation_dataset"]),
        num_workers=0,
        batch_size=1
    )

    model = initialize_config(config["model"])
    # print(model)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=config["optimizer"]["lr"],
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"])
    )

    loss_function = initialize_config(config["loss_function"])

    trainer_class = initialize_config(config["trainer"], pass_args=False)

    trainer = trainer_class(
        config=config,
        resume=resume,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=valid_dataloader
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Wave-U-Net for Speech Enhancement")
    parser.add_argument("-C", "--configuration", default= "/home/dsp/dspfile1/bwe/wj/config/dpcrnopus13.2k/dpcrnopus13.2k.json", type=str, help="Configuration (*.json).")
    parser.add_argument("-R", "--resume", action="store_true", help="Resume experiment from latest checkpoint.")
    args = parser.parse_args()

    configuration = json5.load(open(args.configuration))
    configuration["experiment_name"], _ = os.path.splitext(os.path.basename(args.configuration))
    configuration["config_path"] = args.configuration

    main(configuration, resume=args.resume)
