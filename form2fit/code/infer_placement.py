# coding=utf-8
#used to show the result of PlacementNet
import os
import sys 
import cv2
import glob
# import imutils

import torch
import torch.nn 
import numpy as np
from PIL import Image

from form2fit import config
from form2fit.code.ml.dataloader import placement
from form2fit.code.ml.models import PlacementNet
from form2fit.code.ml.dataloader import get_corr_loader
from form2fit.code.utils import ml, misc

import matplotlib.pyplot as plt
# from get_align_img import initial_camera,get_curr_image
import argparse
import warnings
# warnings.filterwarnings("ignore")
# sys.path.append('/usr/local/lib/python3.6/pyrealsense2')


def findcoord(img, threshold=185):
    #_ , thresh1 = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)             # 根据阈值调整范围
    
    thresh = cv2.GaussianBlur(img, (35, 31), 0)                                 # 高斯模糊，去除周边杂物
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
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imwrite('out.jpg', img)

# python form2fit/code/infer_placement.py --weights form2fit/code/ml/savedmodel/epoch180.pth
if __name__ == "__main__":
    def str2bool(s):
        return s.lower() in ["1", "true"]
    parser = argparse.ArgumentParser(description="Descriptor Network Visualizer")
    parser.add_argument("--modelname", default="black-floss", type=str)
    parser.add_argument("--batchsize", type=int, default=4, help="The batchsize of the dataset.")
    parser.add_argument("--dtype", type=str, default="valid")
    parser.add_argument("--num_desc", type=int, default=64)
    parser.add_argument("--num_channels", type=int, default=2)
    parser.add_argument("--background_subtract", action='store_true', help="(bool) Whether to apply background subtract.")
    parser.add_argument("--augment", type=str2bool, default=False)
    parser.add_argument("--root", type=str, default="", help="the path of project")
    parser.add_argument("--weights", type=str, default="form2fit/code/ml/savedmodel/place_epoch20.pth", help="the path of dataset")
    parser.add_argument("--use_official",action = 'store_true' , help ="whether to use official dataloader" )
    opt = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = opt.modelname

    #initilize model
    print("-------------loading model---------------")
    
    model = PlacementNet(num_channels=2)

    #load weights
    state_dict = torch.load(os.path.join(opt.weights), map_location=device)
    model.load_state_dict({k.replace('module.',''):v for k,v in state_dict.items()})
    model.to(device)

    model.eval()

    #inference
    print("--------------getting img----------------")
    
    if 1: # opt.use_official:        # 官方dataloader
        test_loader = placement.get_placement_loader(opt.root, dtype="test", batch_size=opt.batchsize, num_channels=2, sample_ratio=1, augment=False, shuffle=True, background_subtract=opt.background_subtract)
        for batch_i, (imgs, labels) in enumerate(test_loader):
            imgs = imgs.to(device)

            with torch.no_grad():
                out = model(imgs)

                for output in enumerate(out):
                    output = output[1] * 5 + 200
                    output = np.array(output.cpu().numpy().astype('uint8')) 
                    output = output[0]
                    
                    findcoord(output, threshold=185)

    # else:                   # 自制dataloader
    #     pipeline, align = initial_camera()
    #     c_height,d_height = get_curr_image(pipeline, align)
    #     root =  os.path.join(config.ml_data_dir, opt.root, "infer","0")
    #     cv2.imwrite(os.path.join(root,"final_color_height.png"),c_height)
    #     cv2.imwrite(os.path.join(root,"final_depth_height.png"),d_height)

    #     test_loader = placement_infer.get_placement_loader(opt.root, dtype="infer", batch_size=opt.batchsize, num_channels=2, sample_ratio=1, augment=False, shuffle=False, background_subtract=opt.background_subtract)
    #     for batch_i, (imgs) in enumerate(test_loader):
    #         imgs = imgs.to(device)

    #         with torch.no_grad():
    #             out = model(imgs)

    #             for output in enumerate(out):
    #                 output = output[1] * 5 + 200
    #                 output = np.array(output.cpu().numpy().astype('uint8')) 
    #                 output = output[0]
                    
    #                 findcoord(output, threshold=195)


        

