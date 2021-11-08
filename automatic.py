# -*- coding:utf-8 -*-
import os
import cv2
import sys
import time
import random
import argparse
import numpy as np

import torch

import Jetson.GPIO as GPIO
sys.path.append(os.path.dirname(__file__))
from port_setup import setup
from form2fit.code.get_align_img import initial_camera,get_curr_image
        
def pump_on():
    # 让2号位工作
    import Jetson.GPIO as GPIO
    GPIO.setmode(GPIO.BOARD)
    pump = 12
    GPIO.setup(pump, GPIO.OUT)
    GPIO.output(pump, GPIO.HIGH)

def pump_off():
    import Jetson.GPIO as GPIO
    GPIO.setmode(GPIO.BOARD)
    pump = 12
    GPIO.setup(pump, GPIO.OUT)
    GPIO.output(pump, GPIO.LOW)

def rand_coords(radius=260):   
    randcoords = []                                                         # 存放生成的坐标
    sqs = []                                                                # 暂存x y 的点值
    for i in range(4):                                                      #随机生成坐标,以及旋转角度，输出[x,y,z], rz
        while True:
            x = random.uniform(180, radius)
            y = random.uniform(-radius, -40)
            sq = (x ** 2) + (y ** 2)
            if len(randcoords) >= 1:
                sq1 = sqs[0]
            else:
                sq1=0
            if len(randcoords) >= 2:
                sq2 = sqs[1]
            else:
                sq2=0
            if len(randcoords) >= 3:
                sq3 = sqs[2]
            else:
                sq3=0
            if sq <= (radius ** 2) and abs(sq - sq1) > 3600 and abs(sq - sq2) > 3600 and abs(sq - sq3) > 3600:
                randomz = int(random.uniform(0, 90))
                sqs.append(sq)
                randcoords.append(((int(x), int(y), 140), randomz))
                break
    return randcoords


def find_coords():                                                           #随机生成选取坐标，(四选一)
    x = [0,1,2,3]
    random.shuffle(x)                                                      
    return x


def world2pixel(worldpiont):                                                #待补充
    time.sleep(1)
    pixel = 0
    return pixel

def judge():                                                #是否被成功吸取（是否重量发生相应变化）
    time.sleep(1)                                                           #待补充
    judgenum = 1
    return judgenum

def coordup(coord):                                         #计算吸取点上方坐标
    coord = coord
    coord[2] = coord[2] + 85
    return coord

def coorddown(coord):                                       #计算吸取点上方坐标
    coord = coord
    coord[2] = coord[2] - 85
    return coord

def auto_collection():
             
    c_height,d_height = get_curr_image(pipeline, align)
    
    cv2.imwrite("assets/color{}.png".format(batch*2 + 0), c_height)
    cv2.imwrite("assets/depth{}.png".format(batch*2 + 0), d_height)

    chosenum = findcoords[batch]
    coord1, name = coordlist[chosenum]                  # 四选一 [x, y, z]三维坐标 coord1
    print("ready to suck {}".format(name))
    time.sleep(2)
    points = []                                         # points收集所有放置（数据集中的吸取）位置   --机械臂坐标系
    camcoord1 = world2pixel(coord1)
    points.append(camcoord1)                            # coord1 三维坐标对应的像素坐标 camcoord1

    coord1a = [coord1[0], coord1[1], coord1[2], rxyz[0], rxyz[1], rxyz[2]]              # 机械臂格式[x,y,z,rx,ry,rz], coord1a
    coord1up = coordup(coord1a)                         # 计算吸取上方位置
    mycobot.send_coords(coord1up,speed=40, mode=1)      # 机械臂准备吸取
    print("::send_coords {}, speed 40\n".format(coord1up))
    time.sleep(3)  
    coord1a = coorddown(coord1up)                               
    mycobot.send_coords(coord1a,speed=20, mode=1)        # 机械臂吸取
    print("::send_coords {}, speed 40\n".format(coord1a))
    time.sleep(3)  
    coord1up = coordup(coord1a)
    mycobot.send_coords(coord1up,speed=20, mode=1)      # 机械臂抬起
    print("::send_coords {}, speed 40\n".format(coord1up))
  
    judgenum = judge()
    if judgenum == 0:
        print("The suction failed! Abord")
        pump_off()
        raise RuntimeError 
    else:
        print("The suction is complete.")
        print(randcoords[batch])
        coord2, rz1= randcoords[batch]                         #随机生成坐标和角度 [x,y,z]coord2  rz1
        camcoord2 = world2pixel(coord2)
        points.append(camcoord2)

        coord2a = [coord2[0], coord2[1], coord2[2], rxyz[0], rxyz[1], rz1]
        
        coord2up = coordup(coord2a)
        time.sleep(3) 
        mycobot.send_coords(coord2up,speed=40, mode=1)      # 机械臂准备放置
        print("::send_coords, coord value {}, speed 70\n".format(coord2up))
        time.sleep(3)
        coord2a = coorddown(coord2up)
        mycobot.send_coords(coord2a,speed=20, mode=1)        # 机械臂放置
        print("::send_coords, coord value {}, speed 70\n".format(coord2))
        time.sleep(2)
        pump_off()                                          # 机械臂松手
        time.sleep(2)
        coord2up = coordup(coord2a)
        mycobot.send_coords(coord2up,speed=20, mode=1)      # 机械臂抬起
        print("::send_coords, coord value {}, speed 70\n".format(coord2up))
        time.sleep(3)
        reset = [103.0, -133.5, 131.22, 51.06, 90.35, -123.57]
        mycobot.send_angles(reset, 30)                      # 机械臂回原位置
        c_height, d_height = get_curr_image(pipeline, align)# 记录图像
        cv2.imwrite("assets/color{}.png".format(batch*2 + 1), c_height)                 # 同为第一时刻的final和第二时刻的init图像
        cv2.imwrite("assets/depth{}.png".format(batch*2 + 1), d_height)
        angles1 = coord2a[5] - coord1a[5]                   #计算旋转角度，并存储
        angles.append(angles1)  

if __name__ == "__main__":
    def str2bool(s):
        return s.lower() in ["1", "true"]
    parser = argparse.ArgumentParser(description="Descriptor Network Visualizer")
    parser.add_argument("--modelname", default="black-floss", type=str)
    parser.add_argument("--batchsize", type=int, default=8, help="The batchsize of the dataset.")
    parser.add_argument("--dtype", type=str, default="valid")
    parser.add_argument("--num_desc", type=int, default=64)
    parser.add_argument("--num_channels", type=int, default=2)
    parser.add_argument("--background_subtract", type=tuple, default=None)
    parser.add_argument("--augment", type=str2bool, default=False)
    parser.add_argument("--epochs", type=int, default=160, help="The number of training epochs.")
    parser.add_argument("--weights", type=str, default="form2fit/code/ml/savedmodel/new150.pth", help="the path of dataset")
    opt = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # all_data_collection

    
    a = ([216.0, 44.4, 132.3], 'cylinder')
    b = ([218.3, 117.9, 132.4], 'cube')
    c = ([287.7, 112.3, 132.1], 'triangle')
    d = ([290.5, 42.4, 133.6], 'prism')
    coordlist = (a, b, c, d)
    rxyz = [177, 0, 0]

    print("---------------initiate-------------")
    pipeline, align = initial_camera()  #初始化D435i相机                   
    mycobot = setup()                           #初始化机械臂
    pump_on()
    time.sleep(3)
    angles = [90, 0, -90, 0, 0, 0]
    angles2 = [90, 0, -115, 0, 0, 0]
    mycobot.send_angles(angles, 40)#调整机械臂初始位置
    print("::send_angles() ==> angles {}, speed 80\n".format(angles))
    time.sleep(3)
    mycobot.send_angles(angles2, 40)
    print("::send_angles() ==> angles {}, speed 60\n".format(angles2))
    time.sleep(2)
    print("Arm has been initiated.")

    for epoch in range(opt.epochs):
        print("-------------epoch{}-----------------".format(epoch))
        findcoords = find_coords()                          # 每个epoch开始时，随机生成选取坐标list，(四选一) 
        randcoords = rand_coords()                          # 随机生成放置的四个坐标list
        for batch in range(4):
            
            auto_collection()   #启动自动收集流程

        print("Epoch {} complete. Now start to recollect the items.".format(epoch))


