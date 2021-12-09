# coding=utf-8
#used to show the result of SuctionNet
import os
import cv2
#import imutils

import torch
import torch.nn 
import numpy as np
from PIL import Image

from form2fit import config
from form2fit.code.ml.dataloader import suction
from form2fit.code.ml.models import SuctionNet


import argparse
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def findcoord(i, img, threshold=185):
    #_ , thresh1 = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)             # 根据阈值调整范围
    
    thresh = cv2.GaussianBlur(img, (25, 25), 0)                                 # 高斯模糊，去除周边杂物
    _ , thresh2 = cv2.threshold(thresh, threshold, 255, cv2.THRESH_BINARY) 
    
    cnts = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] 

    for c in cnts:                                                                  

        M = cv2.moments(c)                                                          # 获取中心点
        if M["m00"] == 0:
            break
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
                                                                                    
        #cv2.drawContours(img, [c], -1, (0, 255, 0), 2)                             # 画出轮廓
        cv2.circle(img, (cX, cY), 7, (0, 0, 0), -1)                                 # 画出中点
        cv2.putText(img, "center", (cX - 20, cY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    image_np = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    
    cv2.imwrite('out'+str(i)+'.jpg', image_np)


# python form2fit/code/infer_suction.py --weights form2fit/code/ml/savedmodel/epoch180.pth
if __name__ == "__main__":
    def str2bool(s):
        return s.lower() in ["1", "true"]
    parser = argparse.ArgumentParser(description="Descriptor Network Visualizer")
    parser.add_argument("--modelname", default="black-floss", type=str)
    parser.add_argument("--batchsize", type=int, default=1, help="The batchsize of the dataset.")
    parser.add_argument("--dtype", type=str, default="valid")
    parser.add_argument("--num_desc", type=int, default=64)
    parser.add_argument("--num_channels", type=int, default=2)
    parser.add_argument("--background_subtract", action='store_true')
    parser.add_argument("--augment", type=str2bool, default=False)
    parser.add_argument("--root", type=str, default="", help="the path of dataset")
    parser.add_argument("--weights", type=str, default="form2fit/code/ml/savedmodel/suc_epoch80.pth", help="the path of dataset")
    opt = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = opt.modelname

    
    print("-------------loading data----------------")
    test_loader = suction.get_suction_loader(opt.root, dtype="test", batch_size=opt.batchsize, num_channels=2, sample_ratio=6, augment=False, shuffle=True, background_subtract=opt.background_subtract)

    #initilize model
    print("-------------loading model---------------")
    #model = CorrespondenceNet(num_channels=2, num_descriptor=64, num_rotations=20)
    model = SuctionNet(num_channels=2)
    model = torch.nn.DataParallel(model)

    #load weights
    state_dict = torch.load(os.path.join(opt.weights), map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    model.eval()

    #inference
    print("-----------infer------------")
    for batch_i, (imgs, labels) in enumerate(test_loader):
        
        imgs = imgs.to(device)
        
        for j in range(len(labels)):
                labels[j] = labels[j].to(device)
        with torch.no_grad():
            out = model(imgs)
            i = 0
            for output in enumerate(out):
                output = output[1] * 5 + 200
                #output = output[1]
                output = np.array(output.cpu().numpy().astype('uint8')) 
                
                #print(output)
                output = output[0]
                #print(output.shape)
                #cv2.imshow("output", output)
                
                #cv2.waitKey(10000)
                findcoord(i, output, threshold=175)
                i += 1

