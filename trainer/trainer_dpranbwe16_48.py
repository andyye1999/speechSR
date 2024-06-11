'''
Author: yehongcen && as_yhc@163.com
Date: 2024-04-23 15:36:50
LastEditors: redust as_yhc@163.com
LastEditTime: 2024-06-07 21:04:47
FilePath: \学位论文代码speechSR\trainer\trainer_dpranbwe16_48.py
Description: 第四章模型训练过程
Copyright (c) 2024 by yehongcen, All Rights Reserved. 
'''
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from trainer.base_trainer_dpranbwe16_48 import BaseTrainer
from util.utils import compute_STOI, compute_PESQ, LSD, LSD_NEW, SI_SDR, SNR
# from src.generator import GeneratorEBEN
# from src.discriminator import DiscriminatorEBENMultiScales
plt.switch_backend('agg')

WARMUP_ITERATIONS = 15000


class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            loss_function,
            optimizer,
            train_dataloader,
            validation_dataloader,
    ):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.train_data_loader = train_dataloader
        self.validation_data_loader = validation_dataloader




    def _train_epoch(self, epoch):
        loss_total = 0.0
        gen_loss_total = 0
        dis_loss_total = 0
        melloss_total = 0
        for i, (mixture, clean, filename) in enumerate(self.train_data_loader):
            # clean = mixture
            mixture = mixture.to(self.device)
            clean = clean.to(self.device)
            input = torch.squeeze(mixture, dim=1)
            target = torch.squeeze(clean, dim=1)
            # max_bone = max_bone.to(self.device)
            # max_air = max_air.to(self.device)

            # mixture = self.model.cut_tensor(mixture)
            # clean = self.model.cut_tensor(clean)
            # train_gen = self.step % 2 == 0
            # train_dsc = self.step % 2 == 1
            # train_dsc = self.step % 2 == 1 self.step >= WARMUP_ITERATIONS
            # train generator
            # enhanced_speech = self.model(mixture)
            for optimizer_idx in [0, 1]: # 生成器和判别器共同训练
                input_stft = torch.stft(input, n_fft=1200, hop_length=600, win_length=1200)  # (Bs, F, T, 2) # STFT
                output_stft = self.model(input_stft) # 通过模型
                out_wav = torch.istft(output_stft, n_fft=1200,
                                      hop_length=600, 
                                      win_length=1200) # ISTFT
                enhanced_speech = torch.unsqueeze(out_wav, dim=1)
                if optimizer_idx == 0: # 训练生成器
                    loss_rec_time = F.l1_loss(enhanced_speech, clean, reduction="mean") # 时域重建损失
                    loss_rec_mel = self.loss2(enhanced_speech, clean) # 频域重建损失
                    loss_G = 100 * loss_rec_time + loss_rec_mel
                    melloss_total += loss_G.item()
                    logits_D_fake, features_D_fake = self.discriminator(enhanced_speech) #判别器通过生成数据
                    logits_D_real, features_D_real = self.discriminator(clean) #判别器通过真实数据

                    # gen_loss = self.loss_function(reference_embeddings, enhanced_embeddings)
                    loss_fm = 0
                    loss_adv = 0
                    for i, scale in enumerate(logits_D_fake):
                        loss_adv += F.relu(1-scale).mean()  # 对抗损失
                    loss_adv /= len(logits_D_fake)
                    for i in range(len(features_D_fake)):
                        for j in range(len(features_D_fake[0])):
                            loss_fm += F.l1_loss(features_D_fake[i][j], features_D_real[i][j].detach(), reduction="mean") / \
                                       (features_D_real[i][j].detach().abs().mean() * (len(features_D_fake[0])-1))
                    loss_fm /= len(features_D_fake)  # 特征损失
                    # self.writer.add_scalar(f"Train/loss_fm", loss_fm, self.step)
                    # self.writer.add_scalar(f"Train/loss_adv", loss_adv, self.step)
                    # lamda = self.compute_ema_lambda_adaptive(loss_fm,loss_adv)
                    lamda = 10
                    loss_G += lamda * loss_fm + loss_adv  # 权重

                    # self.writer.add_scalar(f"Train/loss_GLoss", loss_G.item(), self.step)
                    # self.writer.add_scalar(f"Train/loss_rec_time", loss_rec_time.item(), self.step)
                    # self.writer.add_scalar(f"Train/commit_loss", commit_loss.item(), self.step)

                    self.writer.add_scalar(f"Train/loss_rec_mel", loss_rec_mel.item(), self.step)

                    gen_loss_total += loss_G.item()

                    # enhanced = self.model(mixture)
                    # loss = self.loss_function(clean, enhanced)
                    self.optimizer.zero_grad()

                    loss_G.backward()
                    self.optimizer.step()
                # train discriminator  训练判别器
                else:
                    self.optimizer2.zero_grad()
                    # enhanced_speech, decomposed_enhanced_speech = self.model(mixture)

                    enhanced_speech_detach = enhanced_speech.detach()

                    logits_D_fake, features_D_fake = self.discriminator(enhanced_speech_detach)
                    logits_D_real, features_D_real = self.discriminator(clean)
                    loss_D = 0
                    loss_D1 = 0
                    loss_D2 = 0
                    for i, scale in enumerate(logits_D_fake):
                        loss_D1 += F.relu(1 + scale).mean()
                    loss_D1 /= len(logits_D_fake)

                    for i, scale in enumerate(logits_D_real):
                        loss_D2 += F.relu(1 - scale).mean()
                    loss_D2 /= len(logits_D_real)
                    loss_D = loss_D1 + loss_D2 # 判别器损失函数
                    dis_loss_total += loss_D.item()
                    loss_D.backward()
                    self.optimizer2.step()
                    # self.writer.add_scalar(f"Train/disLoss", loss_D.item(), self.step)
                    # self.writer.add_scalar(f"Train/loss_D1", loss_D1, self.step)
                    # self.writer.add_scalar(f"Train/loss_D2", loss_D2, self.step)

            self.step += 1

        dl_len = len(self.train_data_loader)
        self.writer.add_scalar(f"Train/gentotalLoss", gen_loss_total / dl_len, epoch)
        self.writer.add_scalar(f"Train/distotalLoss", dis_loss_total / dl_len, epoch)
        self.writer.add_scalar(f"Train/melloss_total", melloss_total / dl_len, epoch)

        # self.writer.add_scalar(f"Train/Loss", loss_total / dl_len, epoch)
    '''
    description: 验证阶段
    return {*}
    '''
    @torch.no_grad()
    def _validation_epoch(self, epoch):
        visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
        visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
        visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]

        sample_length = self.validation_custom_config["sample_length"]

        stoi_c_n = []  # clean and noisy
        stoi_c_e = []  # clean and enhanced
        pesq_c_n = []
        pesq_c_e = []
        lsd_c_n = []
        lsd_c_e = []
        lsd_HIGH_n = []
        lsd_HIGH_e = []
        lsd_LOW_n = []
        lsd_low_e = []
        snr_c_n = []
        snr_c_e = []
        sisdr_c_n = []
        sisdr_c_e = []




        for i, (mixture, clean, name ) in enumerate(self.validation_data_loader):
            assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
            name = name[0]
            padded_length = 0
            # clean = mixture
            mixture = mixture.to(self.device)  # [1, 1, T]
            clean = clean.to(self.device)
            # max_bone = max_bone.to(self.device)
            # max_air = max_air.to(self.device)
            # The input of the model should be fixed length.
            if mixture.size(-1) % sample_length != 0: # 整除
                padded_length = sample_length - (mixture.size(-1) % sample_length)
                mixture = torch.cat([mixture, torch.zeros(1, 1, padded_length, device=self.device)], dim=-1)

            assert mixture.size(-1) % sample_length == 0 and mixture.dim() == 3
            mixture_chunks = list(torch.split(mixture, sample_length, dim=-1))

            # enhanced_chunks = []
            # for chunk in mixture_chunks:
            #     # output = self.model(chunk)
            #     # output = output * max_bone
            #     # enhanced_chunks.append(self.model(chunk).detach().cpu())
            #     chunk = torch.squeeze(chunk, dim=1)
            #     input_stft = torch.stft(chunk, n_fft=1200, hop_length=600, win_length=1200)  # (Bs, F, T, 2)
            #
            #     output_stft = self.model(input_stft)
            #     out_wav = torch.istft(output_stft, n_fft=1200,
            #                           hop_length=600,
            #                           win_length=1200)
            #     output = torch.unsqueeze(out_wav, 1)
            #     enhanced_chunks.append(output)
            # enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T]
            chunk = torch.squeeze(mixture, dim=1)
            input_stft = torch.stft(chunk, n_fft=1200, hop_length=600, win_length=1200)  # (Bs, F, T, 2) STFT

            output_stft = self.model(input_stft) # 经过模型
            out_wav = torch.istft(output_stft, n_fft=1200,
                                  hop_length=600,
                                  win_length=1200) # ISTFT
            enhanced = torch.unsqueeze(out_wav, 1)
            if padded_length != 0: # 截断
                enhanced = enhanced[:, :, :-padded_length]
                mixture = mixture[:, :, :-padded_length]

            # clean = clean * max_bone
            # mixture = mixture * max_bone
            enhanced = enhanced.detach().cpu()
            enhanced = enhanced.reshape(-1).numpy()
            clean = clean.cpu().numpy().reshape(-1)
            mixture = mixture.cpu().numpy().reshape(-1)

            assert len(mixture) == len(enhanced) == len(clean)

            # Visualize audio
            if i <= visualize_audio_limit:
                self.writer.add_audio(f"Speech/{name}_Noisy", mixture, epoch, sample_rate=48000)
                self.writer.add_audio(f"Speech/{name}_Enhanced", enhanced, epoch, sample_rate=48000)
                self.writer.add_audio(f"Speech/{name}_Clean", clean, epoch, sample_rate=48000)

            # Visualize waveform
            if i <= visualize_waveform_limit:
                fig, ax = plt.subplots(3, 1)
                for j, y in enumerate([mixture, enhanced, clean]):
                    ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                        np.mean(y),
                        np.std(y),
                        np.max(y),
                        np.min(y)
                    ))
                    librosa.display.waveshow(y, sr=48000, ax=ax[j])
                plt.tight_layout()
                self.writer.add_figure(f"Waveform/{name}", fig, epoch)

            # Visualize spectrogram 可视化频谱
            noisy_mag, _ = librosa.magphase(librosa.stft(mixture, n_fft=2048, hop_length=512, win_length=2048))
            enhanced_mag, _ = librosa.magphase(librosa.stft(enhanced, n_fft=2048, hop_length=512, win_length=2048))
            clean_mag, _ = librosa.magphase(librosa.stft(clean, n_fft=2048, hop_length=512, win_length=2048))

            if i <= visualize_spectrogram_limit:
                fig, axes = plt.subplots(3, 1, figsize=(6, 6))
                for k, mag in enumerate([
                    noisy_mag,
                    enhanced_mag,
                    clean_mag,
                ]):
                    axes[k].set_title(f"mean: {np.mean(mag):.3f}, "
                                      f"std: {np.std(mag):.3f}, "
                                      f"max: {np.max(mag):.3f}, "
                                      f"min: {np.min(mag):.3f}")
                    librosa.display.specshow(librosa.amplitude_to_db(mag), cmap="magma", y_axis="linear", ax=axes[k], sr=48000)
                plt.tight_layout()
                self.writer.add_figure(f"Spectrogram/{name}", fig, epoch)

            # Metric
            stoi_c_n.append(compute_STOI(clean, mixture, sr=48000)) # 计算STOI
            stoi_c_e.append(compute_STOI(clean, enhanced, sr=48000))
            # pesq_c_n.append(compute_PESQ(clean, mixture, sr=48000))
            # pesq_c_e.append(compute_PESQ(clean, enhanced, sr=48000))
            hf = int(1025 * (48000 / 16000))
            lsd_j= LSD_NEW(enhanced, clean) # 计算LSD
            base_lsd_j= LSD_NEW(mixture, clean)
            lsd_c_n.append(base_lsd_j)
            lsd_c_e.append(lsd_j)
            snr_j = SNR(enhanced,clean)
            base_snr_j = SNR(mixture,clean) # 计算SNR
            snr_c_n.append(base_snr_j)
            snr_c_e.append(snr_j)
            sisdr_j = SI_SDR(enhanced,clean) # 计算SI-SNR
            base_sisdr_j = SI_SDR(mixture,clean)
            sisdr_c_n.append(base_sisdr_j)
            sisdr_c_e.append(sisdr_j)


        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)
        self.writer.add_scalars(f"Metric/STOI", {
            "Clean and noisy": get_metrics_ave(stoi_c_n),
            "Clean and enhanced": get_metrics_ave(stoi_c_e)
        }, epoch)
        # self.writer.add_scalars(f"Metric/PESQ", {
        #     "Clean and noisy": get_metrics_ave(pesq_c_n),
        #     "Clean and enhanced": get_metrics_ave(pesq_c_e)
        # }, epoch)
        self.writer.add_scalars(f"Metric/LSD", {
            "Clean and noisy": get_metrics_ave(lsd_c_n),
            "Clean and enhanced": get_metrics_ave(lsd_c_e)
        }, epoch)
        self.writer.add_scalars(f"Metric/SNR", {
            "Clean and noisy": get_metrics_ave(snr_c_n),
            "Clean and enhanced": get_metrics_ave(snr_c_e)
        }, epoch)
        self.writer.add_scalars(f"Metric/SISDR", {
            "Clean and noisy": get_metrics_ave(sisdr_c_n),
            "Clean and enhanced": get_metrics_ave(sisdr_c_e)
        }, epoch)


        score = 1 / get_metrics_ave(lsd_c_e)
        return score
