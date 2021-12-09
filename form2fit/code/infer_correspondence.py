# coding=utf-8
#used to show the result of CorrespondenceNet
import os
import sys 
import cv2
import time

import torch
import torch.nn 
import numpy as np
from PIL import Image

from MulticoreTSNE import MulticoreTSNE as TSNE
# from sklearn.manifold import TSNE

from form2fit import config
from form2fit.code.ml.dataloader import correspondence, correspondence_infer
from form2fit.code.ml.models import CorrespondenceNet
# from form2fit.code.utils import ml, misc

import matplotlib.pyplot as plt
# from get_align_img import initial_camera,get_curr_image
import argparse
import warnings
# warnings.filterwarnings("ignore")
# sys.path.append('/usr/local/lib/python3.6/pyrealsense2')
def tsne_visualization(data):
    
    print("start tSNE visualizaion")
    shape = data.shape
    data = data.transpose(1, 2, 0).reshape(-1, 64)
    print("init tsne")
    # tsne
    # tsne = TSNE(n_components=3,init='pca', random_state=0)
    start_time = time.time()
    tsne = TSNE(n_components=3,n_jobs=16)
    print("start transform")
    result = tsne.fit_transform(data)
    consume = (time.time()-start_time)/60
    print ("cosume -time:",consume )
    print(result.shape, type(result), result[0, 2])

    # 每个轴上的坐标都归一化到[0,1]
    result_min, result_max = np.min(result, 0), np.max(result, 0)
    result = (result - result_min) / (result_max - result_min)

    result = result.reshape(shape[-2],shape[-1],-1)
    assert result.shape[2] == 3
    # cv2.imshow('t-SNE result',result)
    cv2.imwrite("tSNE_result_multicore.png",result)
    # cv2.waitKey()
    return result


def save_as_large_vis_txt(dir,filename,data):
    filedir = os.path.join(dir,filename)

    
    embeddings = data.transpose(1, 2, 0).reshape(-1, 64)
    
    N = len(embeddings)
    first_line = np.array([[N,64]],dtype = int)
    first_line = str(N)+" "+str(64)+"\n"
    with open(filedir, 'w') as f:
        f.write(first_line)
        # np.savetxt(f,first_line,delimiter=" ")
        np.savetxt(f,embeddings,delimiter=" ")


if __name__ == "__main__":
    def str2bool(s):
        return s.lower() in ["1", "true"]
    parser = argparse.ArgumentParser(description="Descriptor Network Visualizer")
    parser.add_argument("--modelname", default="black-floss", type=str)
    parser.add_argument("--batchsize", type=int, default=1, help="The batchsize of the dataset.")

    parser.add_argument("--sample_ratio",type = int,default=5, help="The number of training epochs." )
    parser.add_argument("--dtype", type=str, default="valid")
    parser.add_argument("--num_desc", type=int, default=64)
    parser.add_argument("--num_channels", type=int, default=2)
    parser.add_argument("--background_subtract", action='store_true', help="(bool) Whether to apply background subtract.")
    parser.add_argument("--augment", type=str2bool, default=False)
    parser.add_argument("--root", type=str, default="fruits", help="the path of project")
    parser.add_argument("--weights", type=str, default="form2fit/code/ml/savedmodel/corres-epoch1680.pth", help="the path of dataset")
    parser.add_argument("-uo","--use_official",action = 'store_true' , help ="whether to use official dataloader" )
    opt = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_descriptor = opt.num_desc
    batch_size = opt.batchsize
    background_subtract = (0.04, 0.047)#opt.background_subtract
    num_channels = 2
    sample_ratio = opt.sample_ratio

    
    #initilize model
    print("-------------loading model---------------")
    
    model = CorrespondenceNet(num_channels=num_channels, 
                            num_descriptor = num_descriptor,
                            num_rotations=20)

    #load weights
    state_dict = torch.load(os.path.join(opt.weights), map_location=device)
    model.load_state_dict({k.replace('module.',''):v for k,v in state_dict.items()})
    model.to(device)

    model.eval()

    #inference
    print("--------------getting img----------------")
    
    if opt.use_official:        # 官方dataloader
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
                        
                        # tsne_visualization(output)
                    
                    for output in out_t:
                        
                        # output = output * 5 + 200
                        output = np.array(output.cpu().numpy().astype('uint8'))
                        save_as_large_vis_txt("../LargeVis","1207corres.txt",output)
                        print('has saved!')
                        # np.savetxt('out_t.txt',output) 
                        # output = output[0]
                        # tsne_visualization(output)

        

