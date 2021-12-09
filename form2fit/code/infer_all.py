# coding=utf-8
#used to show the result of CorrespondenceNet
import os
import sys 
import cv2
import time
# import glob
# import imutils

import torch
import torch.nn 
import numpy as np
from PIL import Image

from form2fit import config
# from form2fit import 
from form2fit.code.ml.dataloader import suction#,suction_infer
from form2fit.code.ml.models import SuctionNet
from form2fit.code.ml.dataloader import placement#,placement_infer
from form2fit.code.ml.models import PlacementNet
from form2fit.code.ml.dataloader import correspondence, correspondence_infer
from form2fit.code.ml.models import CorrespondenceNet
from form2fit.code.planner.planner import Planner

# from form2fit.code.utils import ml, misc


import matplotlib.pyplot as plt
# from get_align_img import initial_camera,get_curr_image
import argparse
import warnings
# warnings.filterwarnings("ignore")
# sys.path.append('/usr/local/lib/python3.6/pyrealsense2')

def findcoord(img, threshold=185,savename='out.jpg'):
    #_ , thresh1 = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)             # 根据阈值调整范围
    
    thresh = cv2.GaussianBlur(img, (55, 51), 0)                                 # 高斯模糊，去除周边杂物
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
    cv2.imwrite(savename, img)

# suction_out = 0
# place_out = 0
# out_s = 0
# out_t = 0
# python form2fit/code/infer_all.py --weights form2fit/code/ml/savedmodel/epoch180.pth
if __name__ == "__main__":
    def str2bool(s):
        return s.lower() in ["1", "true"]
    parser = argparse.ArgumentParser(description="Descriptor Network Visualizer")
    parser.add_argument("--modelname", default="black-floss", type=str)
    parser.add_argument("--batchsize", type=int, default=1, help="The batchsize of the dataset.")
    parser.add_argument("--sample_ratio",type = int,default=1, help="The number of training epochs." )
    parser.add_argument("--dtype", type=str, default="valid")
    parser.add_argument("--num_desc", type=int, default=64)
    parser.add_argument("--num_channels", type=int, default=2)
    parser.add_argument("--num_rotations",type=int,default=20,help="number of rorations used to divide 360° equally")
    parser.add_argument("--background_subtract", action='store_true', help="(bool) Whether to apply background subtract.")
    parser.add_argument("-a","--augment", type=str2bool, default=False)
    parser.add_argument("--root", type=str, default="fruits", help="the name of dataset")
    parser.add_argument("-sw","--suction_weights", type=str, default="form2fit/code/ml/savedmodel/suc_debug.pth", help="the weight of suction nework")
    parser.add_argument("-pw","--place_weights", type=str, default="form2fit/code/ml/savedmodel/place_debug.pth", help="the weight of placement nework")
    parser.add_argument("-cw","--corres_weights", type=str, default="form2fit/code/ml/savedmodel/final.pth", help="the weight of correspondence nework")
    parser.add_argument("-uo","--use_official",action = 'store_true' , help ="whether to use official dataloader" )
    parser.add_argument("-vis","--need_process_vis",action = 'store_true',help = "whether to visualization three network's output")
    opt = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_descriptor = opt.num_desc
    batch_size = opt.batchsize
    background_subtract = (0.04, 0.047)#opt.background_subtract
    num_channels = 2
    sample_ratio = opt.sample_ratio
    need_process_vis = opt.need_process_vis
    use_official = opt.use_official
    num_rotations = opt.num_rotations
    #==========suction network=============
    
    print("-------------loading suction model---------------")
    
    model = SuctionNet(num_channels=2)

    #load weights
    state_dict = torch.load(os.path.join(opt.suction_weights), map_location=device)
    model.load_state_dict({k.replace('module.',''):v for k,v in state_dict.items()})
    model.to(device)

    model.eval()

    #inference
    print("--------------getting img----------------")
    
    if use_official:        # 官方dataloader
        print("get in use_official")
        test_loader = suction.get_suction_loader(opt.root, 
                                                dtype="test",
                                                batch_size=opt.batchsize,
                                                num_channels=2,
                                                sample_ratio=1,
                                                augment=False, 
                                                shuffle=False, 
                                                background_subtract=opt.background_subtract)
        for batch_i, (imgs, labels) in enumerate(test_loader):
            imgs = imgs.to(device)

            with torch.no_grad():
                suction_out = model(imgs)
                
                if need_process_vis:
                    for output in suction_out:
                        output = output* 5 + 200
                        output = np.array(output.cpu().numpy().astype('uint8')) 
                        output = output[0]
                        findcoord(output, threshold=195,savename='suction.jpg')

                          # else:                   # 自制dataloader
    #     # pipeline, align = initial_camera()
    #     # c_height,d_height = get_curr_image(pipeline, align)
    #     # root =  os.path.join(config.ml_data_dir, opt.root, "infer","0")
    #     # cv2.imwrite(os.path.join(root,"final_color_height.png"),c_height)
    #     # cv2.imwrite(os.path.join(root,"final_depth_height.png"),d_height)

    #     test_loader = suction_infer.get_suction_loader(opt.root, 
    #                                                     dtype="infer", 
    #                                                     batch_size=opt.batchsize, 
    #                                                     num_channels=2, 
    #                                                     sample_ratio=1, 
    #                                                     augment=False, 
    #                                                     shuffle=False, 
    #                                                     background_subtract=opt.background_subtract)
    #     for batch_i, (imgs) in enumerate(test_loader):
    #         imgs = imgs.to(device)


    #==========placement network=============
    print("-------------loading model---------------")
    
    model = PlacementNet(num_channels=2)

    #load weights
    state_dict = torch.load(os.path.join(opt.place_weights), map_location=device)
    model.load_state_dict({k.replace('module.',''):v for k,v in state_dict.items()})
    model.to(device)

    model.eval()

    #inference
    print("--------------getting img----------------")
    
    if use_official:        # 官方dataloader
        test_loader = placement.get_placement_loader(opt.root, 
                                                    dtype="test", 
                                                    batch_size=opt.batchsize, 
                                                    num_channels=2, 
                                                    sample_ratio=1, 
                                                    augment=False, 
                                                    shuffle=False, 
                                                    background_subtract=opt.background_subtract)
        for batch_i, (imgs, labels) in enumerate(test_loader):
            imgs = imgs.to(device)

    # else:                   # 自制dataloader
    #     # pipeline, align = initial_camera()
    #     # c_height,d_height = get_curr_image(pipeline, align)
    #     # root =  os.path.join(config.ml_data_dir, opt.root, "infer","0")
    #     # cv2.imwrite(os.path.join(root,"final_color_height.png"),c_height)
    #     # cv2.imwrite(os.path.join(root,"final_depth_height.png"),d_height)

    #     test_loader = placement_infer.get_placement_loader(opt.root, 
    #                                                         dtype="test", 
    #                                                         batch_size=opt.batchsize, 
    #                                                         num_channels=2, 
    #                                                         sample_ratio=1, 
    #                                                         augment=False, 
    #                                                         shuffle=False, 
    #                                                         background_subtract=opt.background_subtract)
    #     for batch_i, (imgs) in enumerate(test_loader):
    #         imgs = imgs.to(device)

            with torch.no_grad():
                place_out = model(imgs)

                if need_process_vis:
                    for output in place_out:
                        output = output* 5 + 200
                        output = np.array(output.cpu().numpy().astype('uint8')) 
                        output = output[0]
                        
                        findcoord(output, threshold=195,savename='placement.jpg')

    #==========corespondence network=============
    #initilize model
    print("-------------loading model---------------")
    
    model = CorrespondenceNet(num_channels=num_channels, 
                            num_descriptor = num_descriptor,
                            num_rotations=num_rotations)

    #load weights
    state_dict = torch.load(os.path.join(opt.corres_weights), map_location=device)
    model.load_state_dict({k.replace('module.',''):v for k,v in state_dict.items()})
    model.to(device)

    model.eval()

    #inference
    print("--------------getting img----------------")
    
    if use_official:        # 官方dataloader
        test_loader = correspondence.get_corr_loader(opt.root, 
                                        dtype="test",
                                        batch_size=batch_size, 
                                        num_channels=num_channels, 
                                        sample_ratio=sample_ratio,  
                                        augment=False, 
                                        shuffle=False, 
                                        background_subtract=background_subtract)
        for batch_i, (imgs, labels,centers) in enumerate(test_loader):
            #test里有几个文件夹就进入几次
            if batch_i == 3:
                imgs = imgs.to(device)

                with torch.no_grad():
                    out_s,out_t = model(imgs,centers[0][0],centers[0][1])

                    #输出显示
                    # for output in out_s:
                    #     # output = output * 5 + 200
                    #     output = np.array(output.cpu().numpy().astype('uint8')) 
                    #     # output = output[0]
                        
                        # findcoord(output, threshold=195)
                        
                        # tsne_visualization(output)
                    if need_process_vis:
                        for output in out_t:
                            
                            # output = output * 5 + 200
                            output = np.array(output.cpu().numpy().astype('uint8'))
                            # save_as_large_vis_txt("../LargeVis","test.txt",output)
                            # np.savetxt('out_t.txt',output) 
                            # output = output[0]
                            # tsne_visualization(output)
  
    #==================Planner==============
    #转为numpy
    suction_out = np.array(suction_out.cpu().numpy().astype('uint8')) 
    place_out = np.array(place_out.cpu().numpy().astype('uint8')) 
    # out_s = np.array(out_s.cpu().numpy().astype('uint8')) 
    # out_t = np.array(out_t.cpu().numpy().astype('uint8')) 
    # suction_out = suction_out.cpu()
    # place_out = place_out.cpu()
    # out_s = out_s.cpu()
    # out_t = out_t.cpu()
    
    planner = Planner(centers[0])  # instantiate planner with kit center
    print(type(suction_out))
    ret = planner.plan(suction_out[0,0], place_out[1,0], out_s[0], out_t)  # feed it suction and place heatmaps and descriptor maps

    best_place_uv = ret['best_place_uv']
    best_suction_uv = ret['best_suction_uv']
    best_rotation_idx = ret['best_rotation_idx']

    print("best_place_uv:",best_place_uv)
    print('best_suction_uv',best_suction_uv)
    print("best_rotation_angle",best_rotation_idx*360/num_rotations)