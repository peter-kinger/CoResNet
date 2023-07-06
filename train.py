#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The framework of traing process
~~~~~
Before python train.py, please ensure running the "python -m visdom.server -port=xxx" where xxx is the port assign in options
"""


import torch
import torch.nn as nn
import time
import numpy as np
import hues
import os
from data import get_dataloader
from model import create_model
from options.train_options import TrainOptions
from utils.visualizer import Visualizer


if __name__ == "__main__":

    train_opt = TrainOptions().parse()

    train_dataloader = get_dataloader(train_opt, isTrain=True)

    dataset_size = len(train_dataloader)

    # 注意光谱响应矩阵的部分
    train_model = create_model(train_opt, train_dataloader.hsi_channels,
                               train_dataloader.msi_channels,
                               train_dataloader.sp_matrix,
                               train_dataloader.sp_range)

    train_model.setup(train_opt)
    visualizer = Visualizer(train_opt, train_dataloader.sp_matrix)

    test_dataloader = get_dataloader(train_opt, isTrain=False)

    total_steps = 0

    """
    这段代码是一个训练循环的代码片段，它使用PyTorch的dataloader从训练数据集中迭代地加载数据并进行训练。
    
    具体来说，这段代码使用了一个for循环来遍历train_dataloader中的所有数据。
    在每次循环中，代码会记录当前的迭代次数（即i），并通过调用time.time()函数记录当前时间，以便在之后计算每次迭代的时间。
    
    然后，代码会更新total_steps和epoch_iter的值，这些值用于跟踪训练过程中已经完成的步骤和迭代次数。
    
    接下来，代码调用visualizer.reset()函数，该函数用于重置可视化工具的状态，以便在接下来的训练迭代中使用。
    
    然后，代码调用train_model.set_input(data, True)函数，该函数用于将数据传递给模型进行训练。
    在这里，data是从train_dataloader中加载的数据，True表示这是训练模式。
    
    接着，代码调用train_model.optimize_joint_parameters(epoch)函数，该函数用于优化模型的参数。
    在这里，epoch是当前的训练轮数。
    
    然后，代码打印出当前的迭代信息，包括迭代次数、当前轮数、以及总的轮数（由train_opt.niter和train_opt.niter_decay指定）。
    
    最后，代码调用train_model.cal_psnr()函数，
    该函数用于计算训练过程中的峰值信噪比（PSNR）指标，并将其添加到train_psnr_list列表中。
    
    """

    for epoch in range(train_opt.epoch_count, train_opt.niter + train_opt.niter_decay + 1):

        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        # 每一次都会清零数值
        train_psnr_list = []

        # train_dataloader里面读取了数值
        for i, data in enumerate(train_dataloader):

            iter_start_time = time.time()
            total_steps += train_opt.batchsize
            epoch_iter += train_opt.batchsize

            # 开始的重置
            visualizer.reset()

            # 在这里实现了

            train_model.set_input(data, True)
            train_model.optimize_joint_parameters(epoch)

            hues.info("[{}/{} in {}/{}]".format(i,dataset_size//train_opt.batchsize,
                                                epoch,train_opt.niter + train_opt.niter_decay))

            train_psnr = train_model.cal_psnr()
            train_psnr_list.append(train_psnr)



            """
            
            完全可以仿照它的写法用tensorboard或者matplotlib等数据
            但是老师告诉我说一半推荐使用训练结果完了之后再进行相关的数据绘制
            """


            """
            这段代码是一个训练过程中的可视化部分，其中包含了以下步骤：

            如果当前 epoch 是 print_freq 的倍数，那么执行以下操作：
            调用 train_model.get_current_losses() 方法获取当前 epoch 的损失值
            计算当前迭代的平均时间 t，并调用 visualizer.print_current_losses() 方法打印当前 epoch 的损失值和迭代时间 t
            如果 train_opt.display_id 大于 0，那么执行以下操作：
            调用 visualizer.plot_current_losses() 方法在图表中绘制当前 epoch 的损失值
            调用 visualizer.display_current_results() 方法在可视化界面上展示当前 epoch 的训练结果，包括训练图像和其对应的生成图像
            调用 visualizer.plot_spectral_lines() 方法绘制当前 epoch 的图像的频谱线，并在可视化界面上展示
            调用 visualizer.plot_psnr_sam() 方法绘制当前 epoch 的图像的 PSNR 和 SAM 值，并在可视化界面上展示
            调用 visualizer.plot_lr() 方法绘制当前学习率的值，并在可视化界面上展示
            """
            if epoch % train_opt.print_freq == 0:
                losses = train_model.get_current_losses()
                t = (time.time() - iter_start_time) / train_opt.batchsize
                visualizer.print_current_losses(epoch, epoch_iter, losses, t)
                if train_opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, train_opt, losses)
                    visualizer.display_current_results(train_model.get_current_visuals(),
                                                       train_model.get_image_name(), epoch, True,
                                                       win_id=[1])

                    visualizer.plot_spectral_lines(train_model.get_current_visuals(), train_model.get_image_name(),
                                                   visual_corresponding_name=train_model.get_visual_corresponding_name(),
                                                   win_id=[2,3])
                    visualizer.plot_psnr_sam(train_model.get_current_visuals(), train_model.get_image_name(),
                                             epoch, float(epoch_iter) / dataset_size,
                                             train_model.get_visual_corresponding_name())

                    visualizer.plot_lr(train_model.get_LR(), epoch)


        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, train_opt.niter + train_opt.niter_decay, time.time() - epoch_start_time))

        # train_model.update_learning_rate(np.mean(np.array(train_psnr_list)[:]))
        train_model.update_learning_rate()


    # 这里是设置保存端元的部分
    train_model.savePSFweight()
    # train_model.save_networks(train_opt.niter + train_opt.niter_decay)
    train_model.saveAbundance()
