'''
Author: yehongcen && as_yhc@163.com
Date: 2024-04-23 15:36:50
LastEditors: redust as_yhc@163.com
LastEditTime: 2024-05-23 15:20:29
FilePath: \lastdance\trainer\base_trainer_seanetbwe16_48.py
Description: 第三章模型训练配置
Copyright (c) 2024 by yehongcen, All Rights Reserved. 
'''
import time
from pathlib import Path

import json5
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from util import visualization
from util.utils import prepare_empty_dir, ExecutionTime
# from src.generator import GeneratorEBEN
# from src.discriminator import DiscriminatorEBENMultiScales,MultiScaleDiscriminator
from model.loss import cntloss,multiloss,tunetloss,generator_loss,discriminator_loss
from soundstream.modules.msstftd import MultiScaleSTFTDiscriminator
from soundstream.losses.mel_reconstruction_loss import SpectralReconstructionLoss
# from model.soundstream1 import SoundStream


'''
description: 基础模型配置
return {*}
'''
class BaseTrainer:
    def __init__(self,
                 config,
                 resume: bool,
                 model,
                 loss_function,
                 optimizer):
        self.n_gpu = torch.cuda.device_count()
        self.device = self._prepare_device(self.n_gpu, cudnn_deterministic=config["cudnn_deterministic"])

        # self.generator = GeneratorEBEN(bands_nbr=4, pqmf_ks=32).to(self.device)
        # self.model = SoundStream(32, 32, 4, 64, [2, 4, 5, 8]).to(self.device)
        # self.discriminator = MultiScaleSTFTDiscriminator(filters=32,n_ffts = [128, 256, 64],hop_lengths = [32, 64, 16],win_lengths = [128, 256, 64]).to(self.device)
        self.discriminator = MultiScaleSTFTDiscriminator(filters=32).to(self.device) # 多尺度STFT判别器
        # weights = torch.load('F:\\yhc\\bone\\generator.ckpt')
        # self.model.load_state_dict(weights)
        self.loss1 = cntloss()  # 没用 之前测试用的
        self.loss2 = SpectralReconstructionLoss(sr=48000, reduction='mean', device=self.device,)  # 多尺度梅尔频谱loss
        self.model = model.to(self.device)
        # self.model = self.generator
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.0003, betas=(0.5, 0.9))
        self.optimizer2 = torch.optim.Adam(params=self.discriminator.parameters(), lr=0.0003, betas=(0.5, 0.9))
        # self.optimizer = self.optimizer1
        self.loss_function = generator_loss()
        self.lambda_adaptive_past = None
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.n_gpu))) # 多卡運行

        # Trainer
        self.epochs = config["trainer"]["epochs"]
        self.save_checkpoint_interval = config["trainer"]["save_checkpoint_interval"]
        self.validation_config = config["trainer"]["validation"]
        self.validation_interval = self.validation_config["interval"]
        self.find_max = self.validation_config["find_max"]
        self.validation_custom_config = self.validation_config["custom"]
        self.step = 0

        # The following args is not in the config file. We will update it if the resume is True in later.
        self.start_epoch = 1
        self.best_score = -np.inf if self.find_max else np.inf
        self.root_dir = Path(config["root_dir"]).expanduser().absolute() / config["experiment_name"]
        self.checkpoints_dir = self.root_dir / "checkpoints"
        self.logs_dir = self.root_dir / "logs"
        prepare_empty_dir([self.checkpoints_dir, self.logs_dir], resume=resume)

        self.writer = visualization.writer(self.logs_dir.as_posix())
        self.writer.add_text(
            tag="Configuration",
            text_string=f"<pre>  \n{json5.dumps(config, indent=4, sort_keys=False)}  \n</pre>",
            global_step=1
        )

        if resume: self._resume_checkpoint()

        print("Configurations are as follows: ")
        print(json5.dumps(config, indent=2, sort_keys=False))

        with open((self.root_dir / f"{time.strftime('%Y-%m-%d-%H-%M-%S')}.json").as_posix(), "w") as handle:
            json5.dump(config, handle, indent=2, sort_keys=False)

        self._print_networks([self.model])
    '''
    description: 加载训练过的模型
    param {*} self
    return {*}
    '''
    def _resume_checkpoint(self):
        """Resume experiment from the latest checkpoint.
        Notes:
            To be careful at the loading. if the model is an instance of DataParallel, we need to set model.module.*
        """
        latest_model_path = self.checkpoints_dir.expanduser().absolute() / "latest_model.tar"
        latest_modeldis_path = self.checkpoints_dir.expanduser().absolute() / "latest_modeldis.tar"
        assert latest_model_path.exists(), f"{latest_model_path} does not exist, can not load latest checkpoint."
        assert latest_modeldis_path.exists(), f"{latest_modeldis_path} does not exist, can not load latest checkpoint."

        checkpoint = torch.load(latest_model_path.as_posix(), map_location=self.device)
        checkpoint1 = torch.load(latest_modeldis_path.as_posix(), map_location=self.device)

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_score = checkpoint["best_score"]
        self.step = checkpoint1["step"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.optimizer2.load_state_dict(checkpoint1["optimizer"])

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])

        if isinstance(self.discriminator, torch.nn.DataParallel):
            self.discriminator.module.load_state_dict(checkpoint1["model"])
        else:
            self.discriminator.load_state_dict(checkpoint1["model"])

        print(f"Model checkpoint loaded. Training will begin in {self.start_epoch} epoch.")

    '''
    description: 保存模型
    param {*} self
    param {*} epoch
    param {*} is_best
    return {*}
    '''
    def _save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint to <root_dir>/checkpoints directory.
        It contains:
            - current epoch
            - best score in history
            - optimizer parameters
            - model parameters
        Args:
            is_best(bool): if current checkpoint got the best score, it also will be saved in <root_dir>/checkpoints/best_model.tar.
        """
        print(f"\t Saving {epoch} epoch model checkpoint...")

        # Construct checkpoint tar package
        state_dict = {
            "epoch": epoch,
            "best_score": self.best_score,
            "optimizer": self.optimizer.state_dict()
        }

        state_dict1 = {
            "epoch": epoch,
            "best_score": self.best_score,
            "optimizer": self.optimizer2.state_dict(),
            "step": self.step
        }

        if isinstance(self.model, torch.nn.DataParallel):  # Parallel
            state_dict["model"] = self.model.module.cpu().state_dict()
        else:
            state_dict["model"] = self.model.cpu().state_dict()

        if isinstance(self.discriminator, torch.nn.DataParallel):  # Parallel
            state_dict1["model"] = self.discriminator.module.cpu().state_dict()
        else:
            state_dict1["model"] = self.discriminator.cpu().state_dict()

        """
        Notes:
            - latest_model.tar:
                Contains all checkpoint information, including optimizer parameters, model parameters, etc. New checkpoint will overwrite old one.
            - model_<epoch>.pth: 
                The parameters of the model. Follow-up we can specify epoch to inference.
            - best_model.tar:
                Like latest_model, but only saved when <is_best> is True.
        """
        torch.save(state_dict, (self.checkpoints_dir / "latest_model.tar").as_posix())
        torch.save(state_dict["model"], (self.checkpoints_dir / f"model_{str(epoch).zfill(4)}.pth").as_posix())
        torch.save(state_dict1, (self.checkpoints_dir / "latest_modeldis.tar").as_posix())
        torch.save(state_dict1["model"], (self.checkpoints_dir / f"modeldis_{str(epoch).zfill(4)}.pth").as_posix())

        if is_best:
            print(f"\t Found best score in {epoch} epoch, saving...")
            torch.save(state_dict, (self.checkpoints_dir / "best_model.tar").as_posix())
            torch.save(state_dict1, (self.checkpoints_dir / "best_modeldis.tar").as_posix())

        # Use model.cpu() or model.to("cpu") will migrate the model to CPU, at which point we need remigrate model back.
        # No matter tensor.cuda() or tensor.to("cuda"), if tensor in CPU, the tensor will not be migrated to GPU, but the model will.
        self.model.to(self.device)
        self.discriminator.to(self.device)
    '''
    description: 判断几个GPU
    return {*}
    '''
    @staticmethod
    def _prepare_device(n_gpu: int, cudnn_deterministic=False):
        """Choose to use CPU or GPU depend on "n_gpu".
        Args:
            n_gpu(int): the number of GPUs used in the experiment.
                if n_gpu is 0, use CPU;
                if n_gpu > 1, use GPU.
            cudnn_deterministic (bool): repeatability
                cudnn.benchmark will find algorithms to optimize training. if we need to consider the repeatability of the experiment, set use_cudnn_deterministic to True
        """
        if n_gpu == 0:
            print("Using CPU in the experiment.")
            device = torch.device("cpu")
        else:
            if cudnn_deterministic:
                print("Using CuDNN deterministic mode in the experiment.")
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            device = torch.device("cuda:0")

        return device
    '''
    description: 判断是否是最佳模型
    param {*} self
    param {*} score
    param {*} find_max
    return {*}
    '''
    def _is_best(self, score, find_max=True):
        """Check if the current model is the best.
        """
        if find_max and score >= self.best_score:
            self.best_score = score
            return True
        elif not find_max and score <= self.best_score:
            self.best_score = score
            return True
        else:
            return False

    @staticmethod
    def _transform_pesq_range(pesq_score):
        """transform PESQ range. From [-0.5 ~ 4.5] to [0 ~ 1].
        """
        return (pesq_score + 0.5) / 5

    @staticmethod
    def _print_networks(nets: list):
        print(f"This project contains {len(nets)} networks, the number of the parameters: ")
        params_of_all_networks = 0
        for i, net in enumerate(nets, start=1):
            params_of_network = 0
            for param in net.parameters():
                params_of_network += param.numel()

            print(f"\tNetwork {i}: {params_of_network / 1e6} million.")
            params_of_all_networks += params_of_network

        print(f"The amount of parameters is {params_of_all_networks / 1e6} million.")

    def _set_models_to_train_mode(self):
        self.model.train()

    def _set_models_to_eval_mode(self):
        self.model.eval()
    '''
    description: 训练
    param {*} self
    return {*}
    '''
    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            print(f"============== {epoch} epoch ==============")
            print("[0 seconds] Begin training...")
            timer = ExecutionTime()

            self._set_models_to_train_mode()
            self.discriminator.train()
            self._train_epoch(epoch)

            if self.save_checkpoint_interval != 0 and (epoch % self.save_checkpoint_interval == 0):
                self._save_checkpoint(epoch)

            if self.validation_interval != 0 and epoch % self.validation_interval == 0:
                print(f"[{timer.duration()} seconds] Training is over. Validation is in progress...")

                self._set_models_to_eval_mode()
                score = self._validation_epoch(epoch)

                if self._is_best(score, find_max=self.find_max):
                    self._save_checkpoint(epoch, is_best=True)

            print(f"[{timer.duration()} seconds] End this epoch.")

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _validation_epoch(self, epoch):
        raise NotImplementedError

    '''
    description: 动态平衡器
    return {*}
    '''
    # the dynamic loss balancer
    
    def compute_ema_lambda_adaptive(self, loss_recons, loss_adv):  # 平衡器
        """Compute a dynamic loss ponderation to minimize lambda*loss_recons+loss_adv
        Args:
            loss_recons: 1-element tensor corresponding to the feature matching loss
            loss_adv: 1-element tensor corresponding to the adversarial loss
        Return:
           lambda_adaptive_past: the adaptive ponderation coefficient
        """

        beta = 0.99

        # compute lambda_adaptive_t
        grads_recons = \
        torch.autograd.grad(outputs=loss_recons, inputs=self.model.last_conv.weight, retain_graph=True)[0]
        grads_adv = \
        torch.autograd.grad(outputs=loss_adv, inputs=self.model.last_conv.weight, retain_graph=True)[0]
        lambda_adaptive_new = torch.norm(grads_adv) / (torch.norm(grads_recons) + 1e-4)

        if self.lambda_adaptive_past is None:
            self.lambda_adaptive_past = lambda_adaptive_new
        else:  # Exponential Moving average
            self.lambda_adaptive_past = beta * self.lambda_adaptive_past + (1 - beta) * lambda_adaptive_new

        self.lambda_adaptive_past = torch.clamp(self.lambda_adaptive_past, 0.0, 1e4).detach()

        return self.lambda_adaptive_past
