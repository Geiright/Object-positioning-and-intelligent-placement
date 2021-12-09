import time
import argparse
import numpy as np

import torch
import torch.nn as nn

from form2fit.code.ml import losses
from form2fit.code.ml.dataloader import placement
from form2fit.code.ml.models.placement import PlacementNet
from form2fit.code.infer_suction import findcoord

import warnings
warnings.filterwarnings("ignore")


# python form2fit/code/train_placement.py 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Form2Fit placement Module")
    parser.add_argument("--batchsize", type=int, default=4, help="The batchsize of the dataset.")
    parser.add_argument("--sample_ratio", type=int, default=50, help="The ratio of negative to positive labels.")
    parser.add_argument("--epochs", type=int, default=80, help="The number of training epochs.")
    parser.add_argument("--radius", type=int, default=6, help="The radius of positive points.")
    parser.add_argument("--augment", action='store_true', help="(bool) Whether to apply data augmentation.")
    parser.add_argument("--background_subtract", action='store_true', help="Whether to apply background subtraction.")
    parser.add_argument("--dtype", type=str, default="valid")
    parser.add_argument("--imgsize", type=list, default=[848,480], help="size of final image.")
    parser.add_argument("--root", type=str, default="", help="the path of project")
    parser.add_argument("--savepath", type=str, default="form2fit/code/ml/savedmodel/", help="the path of saved models")
    parser.add_argument("--num_workers", default=1, type=int)
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = opt.batchsize
    epochs = opt.epochs
    root = ""
    savepath = opt.savepath
    #augment = opt.augment
    background_subtract = opt.background_subtract
    if background_subtract:
        print("using background subtraction")
    num_channels = 2
    opentest = 1
    sample_ratio = opt.sample_ratio
    radius = opt.radius

    print("--------------start preparing data--------------")
    

    train_loader = placement.get_placement_loader(root, 
                                        dtype="train", 
                                        batch_size=batch_size, 
                                        num_channels=num_channels, 
                                        sample_ratio=sample_ratio, 
                                        augment=True if opt.augment else False,
                                        shuffle=True,
                                        background_subtract=background_subtract,
                                        radius = radius)
    test_loader = placement.get_placement_loader(root, 
                                        dtype="test", 
                                        batch_size=batch_size, 
                                        num_channels=num_channels, 
                                        sample_ratio=sample_ratio, 
                                        augment=True if opt.augment else False,
                                        shuffle=False,
                                        background_subtract=background_subtract,
                                        radius = radius)

    model = PlacementNet(num_channels=num_channels).to(device)
    model = nn.DataParallel(model)

    criterion = losses.PlacementLoss(sample_ratio=sample_ratio, device=device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4,betas=(0.9,0.999),weight_decay=3e-6)
    train_loss = []
    valid_loss = []
    train_epochs_loss = []
    valid_epochs_loss = []

    print("----------------start training-----------------")
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        train_epoch_loss = []

        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            
            for j in range(len(labels)):
                labels[j] = labels[j].cuda()
            output = model(imgs)
            optimizer.zero_grad()
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())
            #print("epoch={}/{},{}/{}of train, loss={}".format(epoch, opt.epochs, i, len(train_loader),loss.item()))
            if i % (len(train_loader)//2) == 0:
                print("epoch = {}/{}, {}/{} of train, loss = {}".format(epoch, opt.epochs, i, len(train_loader),loss.item()))
                train_epochs_loss.append(np.average(train_epoch_loss))
        if epoch % 10 == 0 and opentest:
            print("------------test-----------")
            for batch_i, (imgst, labelt) in enumerate(test_loader):
                imgst = imgst.to(device)
                with torch.no_grad():
                    outts = model(imgst)
                    i = 0
                    for outt in enumerate(outts):
                        outt = outt[1] * 5 + 200
                        outt = np.array(outt.cpu().numpy().astype('uint8')) 
                        outt = outt[0]
                        findcoord(i, outt, threshold=191)
                        i += 1

        if (epoch % 20 == 0 and epoch > 0) or (epoch < 165 and epoch > 145 and epoch % 5 == 0) or epoch == 75:
                                                                                                    # 140 - 160 精确选取，因为150epoch时达到最佳效果
            print("---------saving model for epoch {}----------".format(epoch))
            savedpath = savepath + 'place_epoch' + str(epoch) + '.pth'
            torch.save(model.state_dict(), savedpath)

        if epoch + 1 == epochs:
            print("---------saving model for the last epoch ----------")
            finalsavedpath = savepath + 'place_final' + '.pth'
            torch.save(model.state_dict(), finalsavedpath)
