import argparse
import time
import os
import torch
import numpy as np
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

import torch.nn as nn

from form2fit.code.ml.dataloader import correspondence
from form2fit.code.ml.models.correspondence import CorrespondenceNet
from form2fit.code.ml import losses
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# python form2fit/code/train_correspondence.py 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Form2Fit suction Module")
    parser.add_argument("--batchsize", type=int, default=2, help="The batchsize of the dataset.")
    parser.add_argument("--sample_ratio", type=int, default=5, help="The ratio of negative to positive labels.")
    parser.add_argument("--epochs", type=int, default=100, help="The number of training epochs.")
    parser.add_argument("-a","--augment", action='store_true', help="(bool) Whether to apply data augmentation.")
    parser.add_argument("--background_subtract", type=tuple, default=(0.04, 0.047), help="apply mask.")
    parser.add_argument("--dtype", type=str, default="valid")
    parser.add_argument("--imgsize", type=list, default=[464,360], help="size of final image.")
    parser.add_argument("--foldername", type=str, default="fruits", help="the name of dataset")
    parser.add_argument("--savepath", type=str, default="form2fit/code/ml/savedmodel/", help="the path of saved models")
    parser.add_argument("-r","--resume",action = "store_true",help = 'train from saved model or from scratch')
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = opt.batchsize
    epochs = opt.epochs
    foldername = opt.foldername
    savepath = opt.savepath
    background_subtract = opt.background_subtract
    num_channels = 2
    sample_ratio = opt.sample_ratio

    print("--------------start preparing data--------------")
    

    train_loader = correspondence.get_corr_loader(foldername=foldername, 
                                        dtype="train", 
                                        batch_size=batch_size, 
                                        num_channels=num_channels, 
                                        sample_ratio=sample_ratio, 
                                        augment=True if opt.augment else False,
                                        shuffle=True,
                                        background_subtract=background_subtract)

   

    model = CorrespondenceNet(num_channels=num_channels, 
                            num_descriptor=64, 
                            num_rotations=20).to(device)
                         
    model = nn.DataParallel(model)

    criterion = losses.CorrespondenceLoss(margin= 8, 
                                        num_rotations= 20,
                                        hard_negative= True,
                                        sample_ratio=sample_ratio,
                                        device=device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4,betas=(0.9,0.999),weight_decay=3e-6)
    train_loss = []
    valid_loss = []
    train_epochs_loss = []
    valid_epochs_loss = []

    if opt.resume:
        path_checkpoint = "form2fit/code/ml/savedmodel/corres-epoch1600.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        model.load_state_dict(checkpoint)
        # model.load_state_dict(checkpoint['net'])  # 加载模型可学习参数

        # optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        # start_epoch = checkpoint['epoch']  # 设置开始的epoch   

    print("----------------start training-----------------")
    t0 = time.time()
    for epoch in range(0,epochs):
        model.train()
        train_epoch_loss = []

        for i, (imgs, labels,centers) in tqdm(enumerate(train_loader)):

            imgs = imgs.to(device)
            # for j in range(len(labels)):#len(labels)==batch_size
            #     labels[j] = labels[j].cuda()
            labels = labels.to(device)
            #centers是list,[[uc,vc]]
            out_s, out_t = model(imgs,centers[0][0],centers[0][1])
            optimizer.zero_grad()
            # print("outs.device",out_s.device)
            # print("outt.device",out_t.device)
            # print("label.device",labels.device)
            loss = criterion(out_s, out_t, labels)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())

            #print("epoch={}/{},{}/{}of train, loss={}".format(epoch, opt.epochs, i, len(train_loader),loss.item()))
            if i % (len(train_loader)//2) == 0:
                print("epoch = {}/{}, {}/{} of train, loss = {}".format(epoch, opt.epochs, i, len(train_loader),loss.item()))
                train_epochs_loss.append(np.average(train_epoch_loss))

        if (epoch % 20 == 0 and epoch != 0): #or (epoch < 300 and epoch > 290):                            # 选择输出的epoch
            print("---------saving model for epoch {}----------".format(epoch))
            savedpath = savepath + 'corres-epoch' + str(epoch) + '.pth'
            torch.save(model.state_dict(), savedpath)

        
        # checkpoint = {
        # "net": model.state_dict(),
        # 'optimizer': optimizer.state_dict(),
        # "epoch": epoch,
        # 'lr_schedule': lr_schedule.state_dict()
        # }



        if epoch + 1 == epochs:
            print("---------saving model for last epoch ----------")
            finalsavedpath = savepath + 'corres-epoch'+str(epochs) + '.pth'
            torch.save(model.state_dict(), finalsavedpath)