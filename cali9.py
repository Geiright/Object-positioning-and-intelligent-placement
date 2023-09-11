from pkgutil import get_data
import time
import random
from turtle import color
import cv2
import numpy as np
from torch import int32
from serialcontrol2 import pump_off
from walle.core import RotationMatrix
from HitbotInterface import HitbotInterface
from form2fit.code.utils import analyse_shape
from form2fit.code.utils import get_center
from form2fit.code.get_align_img import initial_camera,get_curr_image

# data最原始9点标定data：
    # imgpoints = [[390,333], [547,339], [546,197], [41,321], [211,336], [39,12], [584,447]]
    # objpoints = [[179.305, 17.204], [185.1776, 94.6872], [115.0382, 96.4722], [162.6492,-144.8213], [173.095,-68.6748], [17.7836, -139.3039], [240.0178, 108.7411]]

# scara机械臂粘定后的data：         #盒子部分，前4个为标准点位
    # imgpoints = [[893,261], [889,431], [1079,428], [1079,264]]
    # objpoints = [[98.8834, 71.6999], [157.0539, 67.6597], [158.7592,132.9691], [102.5297,135.2645]]
                                    # 物块部分
    # imgpoints = [[563,615], [381,619], [214,623], [211,477], [370,469], [545,461], [230,225]]
    # objpoints = [[214.4791, -46.4357], [212.3297, -105.9849], [210.5402, -159.6319], [162.6528, -159.6316], [162.6527, -108.4162], [162.6526, -50.669], [80.1705, -150.8045]]
def rotz2angle(rotz):
    """Extracts z-rotation angle from rotation matrix.

    Args:
        rotz: (ndarray) The (3, 3) rotation about z.
    """
    return np.arctan2(rotz[1, 0], rotz[0, 0])


def clip_uv(uv, rows, cols):
    """Ensures pixel coordinates are within image bounds.
    """
    uv[:, 0] = np.clip(uv[:, 0], 0, rows - 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, cols - 1)
    return uv

def rotate_uv(uv, angle, rows, cols, cxcy=None):
    """Finds the value of a pixel in an image after a rotation.

    Args:
        uv: (ndarray) The [u, v] image coordinates.
        angle: (float) The rotation angle in degrees.
    """
    txty = [cxcy[0], cxcy[1]] if cxcy is not None else [(rows // 2), (cols // 2)]
    txty = np.asarray(txty)
    uv = np.array(uv)
    aff_1 = np.eye(3)
    aff_3 = np.eye(3)
    aff_1[:2, 2] = -txty
    aff_2 = RotationMatrix.rotz(np.radians(angle))
    aff_3[:2, 2] = txty
    affine = aff_3 @ aff_2 @ aff_1
    affine = affine[:2, :]
    uv_rot = (affine @ np.hstack((uv, np.ones((len(uv), 1)))).T).T
    uv_rot = np.round(uv_rot).astype("int")
    uv_rot = clip_uv(uv_rot, rows, cols)
    return uv_rot


def c2b(M, cam_points):                             # camera  to  board
    assert cam_points.shape == (1,2)
    assert M.shape == (2,3)
    R = M[:,:2]
    T = M[:,2]
    cam_points = np.float32(cam_points)
    board_points = (M @ np.hstack((cam_points, np.ones((len(cam_points), 1)))).T).T
    return board_points

def b2c(Mn, board_points):                          # board  to  camera
    assert Mn.shape == (2,3)
    board_points = np.array(board_points, dtype='float32')
    board_points = np.expand_dims(board_points, axis=0)
    # Mn = np.linalg.inv(M) # 求逆
    cam_points = (Mn @ np.hstack((board_points, np.ones((len(board_points), 1)))).T).T
    return cam_points.tolist()

def get_data_txt(dir='data.txt'):                   # 按照 [imgpoints0, imgpoints1, objpoints0, objpoints1, objpoints2] 的原则读取txt.
    f = open(dir, 'r')
    imgpoints = []
    objpoints = []
    for lines in f:
        ls = lines.strip('\n').replace(' ','').replace('、','/').replace('?','').split(',')
        print("img/n", imgpoints)
        print("obj/n", objpoints)
        imgpoints.append([float(ls[0]), float(ls[1])])
        objpoints.append([float(ls[2]), float(ls[3]), float(ls[4])])
    return imgpoints, objpoints

# [482,261], [488,346], [492,470], [492,537], []
# [95.8741, -68.0953], [125.6157, -67.1776], [168.4202, -67.1775], [191.0811, -67.6724], []
def getM():                                         # 物体部分的坐标转换
    # imgpoints = [[260.0, 170.0], [261.0, 207.0], [263.0, 245.0], [264.0, 284.0], [266.0, 322.0], [268.0, 360.0], [268.0, 399.0], [270.0, 436.0], [300.0, 169.0], [301.0, 207.0], [303.0, 245.0], [303.0, 283.0], [304.0, 321.0], [306.0, 359.0], [308.0, 397.0], [310.0, 435.0], [196.0, 278.0], [339.0, 205.0], [340.0, 243.0], [340.0, 282.0], [342.0, 320.0], [345.0, 358.0], [345.0, 396.0], [346.0, 434.0]]
    # objpoints =  [[90.0, -100.0], [110.0, -100.0], [130.0, -100.0], [150.0, -100.0], [170.0, -100.0], [190.0, -100.0], [210.0, -100.0], [230.0, -100.0], [90.0, -80.0], [110.0, -80.0], [130.0, -80.0], [150.0, -80.0], [170.0, -80.0], [190.0, -80.0], [210.0, -80.0], [230.0, -80.0], [90.0, -60.0], [110.0, -60.0], [130.0, -60.0], [150.0, -60.0], [170.0, -60.0], [190.0, -60.0], [210.0, -60.0], [230.0, -60.0]]
    
    imgpoints, objpoints = get_data_txt()
    
    imgpoints = np.array(imgpoints,dtype='float32')
    objpoints = np.array(objpoints,dtype='float32')
    M, _ = cv2.estimateAffine2D(imgpoints, objpoints,True)      # 第二种方式：九点标定法
    Mn, _ = cv2.estimateAffine2D(objpoints, imgpoints,True)
    return M, Mn                                         # M是从像素坐标系转换为机械臂坐标系的矩阵，Mn反之

def getM_box():                                     # 盒子部分的坐标转换
    imgpoints = [[893,261], [889,431], [1079,428], [1079,264]]
    objpoints = [[98.8834, 71.6999], [157.0539, 67.6597], [158.7592,132.9691], [102.5297,135.2645]]
    imgpoints = np.array(imgpoints,dtype='float32')
    objpoints = np.array(objpoints,dtype='float32')
    M, _ = cv2.estimateAffine2D(imgpoints, objpoints,True)      # 第二种方式：九点标定法
    Mn, _ = cv2.estimateAffine2D(objpoints, imgpoints,True)
    return M, Mn                                         # M是从像素坐标系转换为机械臂坐标系的矩阵，Mn反之



def rand_coords(epoch=100, radius=4200):            # 生成随机的坐标值，主要用于自动标定过程。
    randcoords = []                                                         # 存放生成的坐标
    # sqs = []                                                                # 暂存x y 的点值
    x1,y1 = 90, -100
    randcoords.append(((int(x1), int(y1), -97), 0))
    i = 0
    while(i<epoch):
        if x1 < 230:
            x1 = x1 + 20
        else:
            y1 = y1 + 20
            x1 = 90
        randcoords.append(((int(x1), int(y1), -97), 0))
        i += 1
    
    print(randcoords)
    # for _ in range(epoch):
    #     x1 = random.uniform(90, 230)
    #     y1 = random.uniform(-180, -40)
    #     randomrz = int(random.uniform(0, 90))                                   # 随机生成坐标,以及旋转角度，输出[x,y,z], rz
    #     randcoords.append(((int(x1), int(y1), -92), randomrz))                                 
    return randcoords



def arm_placement(robot, coord2, rz1=0, hand=-1):   # 机械臂的放置指令。hand为-1，则为物体部分。1为盒子部分
    box_pos = [-64.6159, 269.41, -39]            # 不遮挡相机拍摄的机械臂位置，每次结束放置后移动至该位置。
    time.sleep(0.5) 
    a = robot.new_movej_xyz_lr(coord2[0], coord2[1], coord2[2] + 40, 0,140,0,hand)    # 机械臂准备放置
    robot.wait_stop()
    print("ready to place: coord value {}, speed 140\n".format(coord2))
    if a == 1: print("moving") 
    else: 
        print("error, code is {}".format(a))
        raise RuntimeError("arm cannot move to the location {}".format(coord2))
    time.sleep(0.25)
    a = robot.new_movej_xyz_lr(coord2[0], coord2[1], coord2[2] + 40, rz1,140,0,hand)    # 机械臂旋转
    robot.wait_stop()
    time.sleep(0.25)
    print("::placing, coord value {}, speed 100\n".format(coord2))
    a = robot.new_movej_xyz_lr(coord2[0], coord2[1], coord2[2]-5 , rz1,100,0,hand)       # 机械臂放置.z轴如果空中放置则-1，多线程放置则-5
    robot.wait_stop()
    if a == 1: print("moving") 
    else: print("error, code is {}".format(a))
    time.sleep(0.5)
    pump_off()                                                                          # 机械臂松手
    time.sleep(0.5)

    print("::send_coords, coord value {}, speed 140\n".format(coord2))
    a = robot.new_movej_xyz_lr(coord2[0], coord2[1], coord2[2] + 40, rz1,140,0,hand)      # 机械臂抬起
    robot.wait_stop()
    if a == 1: print("moving") 
    else: print("error, code is {}".format(a))
    time.sleep(0.5)
    
    a = robot.new_movej_xyz_lr(box_pos[0], box_pos[1], -34, 0,120,0,1)                   # 机械臂回原位置
    robot.wait_stop()    

def arm_suction(robot, coord2, rz2=0):              # 机械臂的吸取指令。
    a = robot.new_movej_xyz_lr(coord2[0], coord2[1], coord2[2]+40, rz2,120,0,-1)      # 机械臂准备吸取  (盒子这边是1，物块那一边是-1)
    robot.wait_stop()
    if a == 1: print("moving") 
    else: print("error, code is {}".format(a))
    time.sleep(0.5)  

    a = robot.new_movej_xyz_lr(coord2[0], coord2[1], coord2[2]-5, rz2, 90, 0,-1)         # 机械臂吸取
    robot.wait_stop()
    if a == 1: 
        print("moving") 
        print("::send_coords, coord value {}, speed 90\n".format(coord2))
    else: print("error, code is {}".format(a)) 
    time.sleep(0.5)  
    # coord1up = coordup(coord1)
    a = robot.new_movej_xyz_lr(coord2[0], coord2[1], coord2[2] + 40, rz2, 110,0,-1)       # 机械臂抬起
    robot.wait_stop()
    if a == 1: print("moving") 
    else: print("error, code is {}".format(a))



def autocali():                                     # 用于机械臂的手动标定。每一次要手动移动机械臂到指定位置。eye to hand 标定法

    box_pos = [-64.6159, 269.41, -39]   # 机械臂原始位置
    choose = 'box'          # 选择的是盒子内还是盒子外的标定    
    z_down = -74            # 机械臂下降到指针贴合标定处的z轴高度.-74为正好在圆柱体的上方
    box_cali = [[220, 50], [220, 100], [220, 150], [160,50], [160, 100], [160,150], [80, 50], [80, 100], [80,150]]           # 盒子内的标定位置集合（机械臂坐标系）
    out_cali = [[220,-140], [220, -100], [220, -60], [160, -140], [160, -100], [160, -60], [100, -140], [100, -100], [100, -60]]           # 盒子外的标定点位集合（机械臂坐标系）


    print("---------------init camera---------------")
    pipeline, align = initial_camera()

    
    print("-------------init the arm----------------")
    robot_id = 18
    robot = HitbotInterface(robot_id)
    robot.net_port_initial()
    time.sleep(0.5)
    print("initial successed")
    ret = robot.is_connect()
    while ret != 1:
        time.sleep(0.1)
        ret = robot.is_connect()
        print(ret)
    ret = robot.initial(3, 180)
    if ret == 1:
        print("robot initial successful")
        robot.unlock_position()
    else:
        print("robot initial failed")
    if robot.unlock_position():
        print("------unlock------")
    time.sleep(0.5)

    
    if robot.is_connect():
        print("robot online")
        a = robot.new_movej_xyz_lr(box_pos[0], box_pos[1], -34, 0, speed=120, roughly=0, lr=1)
        robot.wait_stop()
        print("robot statics is {}".format(a))
        if a == 1: print("the robot is ready for the collection.")
        time.sleep(0.5)

    if choose == 'box':
        cali = box_cali
    else:
        cali = out_cali
    for i in range(len(cali)):
        coord1 = box_cali[i]
        a = robot.new_movej_xyz_lr(coord1[0], coord1[1], z_down+60, 150,140,0,-1)      # 机械臂准备下降, 手系为-1，盒子方向的手系
        robot.wait_stop()
        print("准备完成。请把物块放置在标定针中央")
        time.sleep(3)
        print("即将下降，请保持物块稳定")
        a = robot.new_movej_xyz_lr(coord1[0], coord1[1], z_down, 60,140,0,-1)      # 机械臂下降, 手系为-1
        robot.wait_stop()
        print("下降完成。请微调")
        time.sleep(2)
        a = robot.new_movej_xyz_lr(coord1[0], coord1[1], z_down+60, 150,140,0,-1)      # 机械臂抬起, 手系为-1
        robot.wait_stop()
        print("第{}个点位标定完成。".format(i))
        a = robot.new_movej_xyz_lr(box_pos[0], box_pos[1], -34, 0, speed=120, roughly=0, lr=1)  # 移回初始位置，进行图像采集
        robot.wait_stop()
        
    print("全部点位标定完成。")


def autocali2():                                    # 用于机械臂的自动标定。每一次让机械臂选择一个位置，放置物块，保存当前的机械臂坐标。
                                                    # 然后机械臂回归原点，相机截取当前图片，二值化取蓝色区域，求蓝色区域质心作为当前的相机坐标系坐标。
    # init position
    box_pos = [-64.6159, 269.41, -39]            # 不遮挡相机拍摄的机械臂位置，每次结束放置后移动至该位置。
    objpoints = []
    imgpoints = []
    allpoints = []                          # allpoints用于存储适合放入神经网络的坐标。格式：[obj0,obj1,obj2,img0,img1]
    epoch = 25
    randcoords = rand_coords(epoch)                          # 随机生成放置的坐标list
    print("---------------init camera---------------")
    pipeline, align = initial_camera()

    print("-------------init the arm----------------")
    robot_id = 18
    robot = HitbotInterface(robot_id)
    robot.net_port_initial()
    time.sleep(0.5)
    print("initial successed")
    ret = robot.is_connect()
    while ret != 1:
        time.sleep(0.1)
        ret = robot.is_connect()
        print(ret)
    ret = robot.initial(3, 180)
    if ret == 1:
        print("robot initial successful")
        robot.unlock_position()
    else:
        print("robot initial failed")
    if robot.unlock_position():
        print("------unlock------")
    time.sleep(0.5)

    if robot.is_connect():
        print("robot online")
        a = robot.new_movej_xyz_lr(box_pos[0], box_pos[1], -34, 0, speed=120, roughly=0, lr=1)
        robot.wait_stop()
        print("robot statics is {}".format(a))
        if a == 1: print("the robot is ready for the collection.")
        time.sleep(0.5)


    ################  开始标定   #################

    
    
    for i in range(epoch): 
        coord2, rz1= randcoords[i]                         #随机生成坐标和角度 [x,y,z]coord2  rz1
        
        arm_placement(robot, coord2, rz1)                               # 机械臂放置，并返回原位置，准备拍摄

        objpoints.append([coord2[0], coord2[1]])
        print("objpoint [{}, {}] no.{}".format(coord2[0], coord2[1], i))
        color_image,_ = get_curr_image(pipeline, align)                                      # 记录图像
        cv2.imwrite("test.jpg", color_image)
        center = get_center.get_center(color_image)                                                       # 求质心
        imgpoints.append(center)
        allpoints.append([center[0], center[1], coord2[0], coord2[1], -98])
        print("imgpoint {} no.{}".format(center, i))
        
        arm_suction(robot, coord2)                               # 机械臂把物体吸起，准备下一回合。
        np.savetxt('data.txt', allpoints, delimiter=',')

    np.savetxt('data.txt', allpoints, delimiter=',')
    print("points saved.")
        
        
        
def deltademo():                                    # 用于机械臂吸取摆放demo实现。
 # way = 2
    # if way == 1:
    # img = cv2.imread("1.png")
    # pts1 = np.float32([[50.4306, -175.9408], [199.0511, -175.1985], [195.3472, 34.5088]]) #[106.8482, 32.2865]
    # pts2 = np.float32([[186,105], [186,516], [764,500] ])#[759,263]     [289,424] [458,426]
    # pts3 = np.float32([[289,424]])
    #     imgpoints = [[11,71], [121.6,72], [231.3,286]]
    #     objpoints = [[31.31, -175.66],[34.31,-116.086], [155.4, -61.19]]
    #     pts1 = np.array(imgpoints, dtype='float32')
    #     pts2 = np.array(objpoints, dtype='float32')
    #     M = cv2.getAffineTransform(pts1, pts2)      # 第一种方式：三点标定法


    # if way == 2:
    #     imgpoints = [[390,333], [547,339], [546,197], [41,321], [211,336], [39,12], [584,447]]
    #     objpoints = [[179.305, 17.204], [185.1776, 94.6872], [115.0382, 96.4722], [162.6492,-144.8213], [173.095,-68.6748], [17.7836, -139.3039], [240.0178, 108.7411]]
    #     imgpoints = np.array(imgpoints,dtype='float32')
    #     objpoints = np.array(objpoints,dtype='float32')
    #     M, _ = cv2.estimateAffine2D(imgpoints, objpoints,True)      # 第二种方式：九点标定法
    #     Mn, _ = cv2.estimateAffine2D(objpoints, imgpoints,True)
    # else: 
    #     print("error: no method specified!")
    #     raise RuntimeError("you need to enter 1 or 2.")
    M, _ = getM()
    cam = np.float32([[375, 167]])
    print(cam.shape)
    print(c2b(M, cam))
 
    

    print("---------------init camera---------------")
    pipeline, align = initial_camera()
    
    print("-------------init the arm----------------")
    robot_id = 18
    robot = HitbotInterface(robot_id)
    robot.net_port_initial()
    time.sleep(0.5)
    print("initial successed")
    ret = robot.is_connect()
    while ret != 1:
        time.sleep(0.1)
        ret = robot.is_connect()
        if(ret):
            print("robot is connected successfully.")
        else:
            time.sleep(2)
    ret = robot.initial(3, 180)
    if ret == 1:
        print("robot initial successful")
        robot.unlock_position()
    else:
        print("robot initial failed")
    if robot.unlock_position():
        print("------unlock------")
    time.sleep(0.5)

    if robot.is_connect():
        print("robot online")
        box_pos = [-64.6159, 269.41, -39]            # 不遮挡相机拍摄的机械臂位置，每次结束放置后移动至该位置。
        a = robot.new_movej_xyz_lr(box_pos[0], box_pos[1], -34, 0, speed=120, roughly=0, lr=1)
        robot.wait_stop()
        print("robot statics is {}".format(a))
        if a == 1: print("the robot is ready for the collection.")
        time.sleep(0.5)

    color_image,_ = get_curr_image(pipeline, align)                                      # 记录图像
    ref_image = cv2.imread("origin_box.jpg")
    cv2.imwrite("test.jpg", color_image)

    obj_list = analyse_shape.get_info_for_arm(color_image, False)             # obj_list [['circle', (278, 194), 0], ['triangle', (335, 324), 78.01278750418338]]
    x, y = analyse_shape.get_kit_offset(color_image, ref_image)                    # 计算盒子的像素平移距离
    x_box = x / 42 * 24                                                         # 计算盒子的机械臂平移距离
    for obj in obj_list:
        center = obj[1]
        rz1 = obj[2]
        if obj[0] == 'circle':
            coord2 = [98.27, 182.8504, -66.39]                  # [98,207] + 24
        elif obj[0] == 'square':
            coord2 = [173.5626, 179.799, -66.39]
        elif obj[0] == 'triangle':
            coord2 = [163.57, 104.0825, -66.39]
        elif obj[0] == 'pentagon':
            coord2 = [95.1082, 104.0858, -66.39]                # coord2代表放置位置
        else:
            raise RuntimeError("Can not identify the item.")
        # coord2[1] += x_box


        print("center is {}".format(center))
        center = np.float32(center)
        center_re = center.reshape(1,2)
        center_arm = c2b(M, center_re)
        print("center_arm is {}".format(center_arm))
        coord1 = [center_arm[0][0], center_arm[0][1], -96]      # coord1代表吸起位置
        
        arm_suction(robot, coord1, rz2=0)                               # 机械臂把物体吸起
        arm_placement(robot, coord2, rz1=-rz1, hand=1)                   # 机械臂把物体放下


if __name__ == '__main__':
   autocali2()
