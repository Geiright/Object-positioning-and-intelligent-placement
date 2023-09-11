import cv2
import numpy as np
import math
import os
import form2fit.config as cfg
from skimage.measure import label
import time
import pickle
'''
分析盒外物体的位置\形状\角度
原理:1.在HSV分割出物体mask
2.根据mask中找到的顶点数量,确定物体的形状,升级:双保险:面积,如果两个得到的结果相同,确定是该形状,若形状不同,用更苛刻的HSV范围重新分割,再次进行第二步,若仍不行,以面积为准
3.位置:直接计算质心,升级:根据上一步的形状生成一个规则的多边形,寻找最贴切的位置,输出规则多边形的中心
4.角度:输出规则多边形的角度
'''
# hf_w = cfg.a_w //2


def remove_small_area(mask, area_th,visual, message):
    '''
        去除小面积连通域
        :param mask: 待处理的mask
        :param area_th: 小于area_th的面积会被消除
        :return: 去除了小面积连通域的mask
    '''
    contours = get_exter_contours(mask, 'simple')
    # contours = get_tree_contours(mask, 'none',-2)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_th:
            cv2.drawContours(mask, [cnt], 0, 0, -1)
    if visual:
        cv2.imshow('after remove small area_{}'.format(message), mask)
    return mask

def remove_big_area(mask, area_th,visual, message):
    '''
        去除小面积连通域
        :param mask: 待处理的mask
        :param area_th: 小于area_th的面积会被消除
        :return: 去除了小面积连通域的mask
    '''
    contours = get_exter_contours(mask, 'simple')
    # contours = get_tree_contours(mask, 'none',-2)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_th:
            cv2.drawContours(mask, [cnt], 0, 0, -1)
    if visual:
        cv2.imshow('after remove big area_{}'.format(message), mask)
    return mask


def remove_inner_white(mask, visual, message):
    contours = get_tree_contours(mask, 'none', -2)
    for cnt in contours:
        cv2.drawContours(mask, [cnt], 0, 0, -1)
    if visual:
        cv2.imshow('after remove inner white_{}'.format(message), mask)
    return mask

def remove_slim(mask, ratio = 10):
    '''
    去掉细长的非0部分,原理:越细长，连通域的面积/周长比就越小
    :param mask:待处理的mask
    :param ratio: ratio=面积/周长
    :return:处理后的mask
    '''
    contours = get_exter_contours(mask, 'none')
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        # print(area)
        # print(perimeter)
        if area < perimeter * ratio:
            cv2.drawContours(mask, [cnt], 0, 0, -1)
    return mask

def largest_cc(mask,bol2img):
    '''
    选出除0像素之外，最大的连通域
    :param mask:一张图
    :param bol2mask
    :return: bool类型的矩阵，true部分对应的就是最大连通域
    '''
    labels = label(mask)
    largest = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    if bol2img:
        largest = np.array(largest).astype('uint8')*255
    return largest

def remove_scattered_pix(mask,th,visual):
    #去除只有th个像素的连通域而不影响其他内容
    labels = label(mask,connectivity=None)
    remove_index = np.where(np.bincount(labels.flat)[1:] <= th)
    for item in remove_index[0]:
        mask = np.where(labels == item +1, 0, mask)
    if visual:
        cv2.imshow('after remove_single_pix', mask)
    return mask

def remove_single_pix_new(mask,visual):
    pass



def mask2bbox(mask):
    #寻找二值化图像的mask=1处的方框
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def remove_surrounding_white(mask,visual):
    '''
    在mask中，去掉贴着图像边缘的白色部分（通常是背景）
    :param mask:处理前的mask
    :param visual: 是否可视化
    :return: mask：处理后的mask
    '''
    h,w = mask.shape
    labels = label(mask)
    if visual:
        cv2.imshow('labels going to remove_surrounding_white',(labels*40).astype('uint8'))
    num = np.max(labels)
    if num > 1:#如果只有一个连通域，不需要处理
        for i in range(num):
            domain = np.where(labels==i+1,1,0)
            if visual:
                cv2.imshow('domain in remove_surrounding_white',(domain.astype('uint8'))*255)
            rmin,rmax,cmin,cmax = mask2bbox(domain)
            if rmin ==0 or rmax == h-1 or cmin == 0 or cmax == w-1:
                labels = np.where(labels == i+1 , 0 , labels)
        mask = np.where(labels != 0,mask,0)
        if visual:
            cv2.imshow('mask in remove_surrounding_white',mask)
    return mask


def remove_inner_black(mask,visual):
    '''
    在mask中去掉白色部分中间的黑色
    :param mask:
    :param visual: 可视化
    :return: mask：处理后的mask
    '''
    h, w = mask.shape
    mask = 255 - mask
    labels = label(mask)
    if visual:
        cv2.imshow('labels going to remove_inner_black', (labels * 40).astype('uint8'))
    num = np.max(labels)
    for i in range(num):
        domain = np.where(labels == i + 1, 1, 0)
        if visual:
            cv2.imshow('domain in remove_inner_black', (domain.astype('uint8')) * 255)
        rmin, rmax, cmin, cmax = mask2bbox(domain)
        if not (rmin == 0 or rmax == h - 1 or cmin == 0 or cmax == w - 1):
            labels = np.where(labels == i + 1, 0, labels)
    mask = np.where(labels != 0, mask, 0)
    mask = 255 - mask
    if visual:
        cv2.imshow('mask in remove_inner_black', mask)
    return mask


def rot_around_point(rot_mtx,value,center_point):
    '''
    批量点的旋转，value的shape是(N,2),其中N表示N个点
    :param rot_mtx: 变换矩阵的内容
    :param value: 要旋转的点
    :param center_point: 旋转的中心点
    :return:
    '''
    trans_mtx_3 = np.eye(3)
    trans_mtx_1 = np.eye(3)
    rot_mtx33 = rot_mtx[:3, :3]
    trans_mtx_3[:2, 2] = center_point
    trans_mtx_1[:2, 2] = -center_point
    trans = trans_mtx_3 @ rot_mtx33 @ trans_mtx_1
    value_ones = np.ones((len(value), 1))
    value = np.hstack((value, value_ones))
    value_after_rot = (trans @ value.T).T
    value_after_rot = value_after_rot[:, :2]
    # assert (value_after_rot[:, 0] <= 464).all()
    # assert (value_after_rot[:, 1] <= 360).all()

    return value_after_rot

def gen_rot_mtx(angle,isdegree):
    '''
    生成逆时针旋转指定角的旋转矩阵
    :param angle:要逆时针旋转的角
    :param isdegree:旋转角的表示方式,如果是用角度表示,此项为True,若是弧度表示,此项为False
    :return:3*3的旋转矩阵
    '''
    if isdegree:
        angle = math.radians(angle)
    cos_rad = math.cos(angle)
    sin_rad = math.sin(angle)
    trans = np.eye(3)
    trans[:2, :2] = [[cos_rad, -sin_rad], [sin_rad, cos_rad]]
    return trans

def gen_rot_mtx_clockwise(angle,isdegree):
    '''
    生成逆时针旋转指定角的旋转矩阵
    :param angle:要逆时针旋转的角
    :param isdegree:旋转角的表示方式,如果是用角度表示,此项为True,若是弧度表示,此项为False
    :return:3*3的旋转矩阵
    '''
    if isdegree:
        angle = math.radians(angle)
    cos_rad = math.cos(angle)
    sin_rad = math.sin(angle)
    trans = np.eye(3)
    trans[:2, :2] = [[cos_rad, sin_rad], [sin_rad, -cos_rad]]
    return trans

def gen_corrs(disassemble, theta, init_point, final_point, init_mask_coord, final_mask_coord, visual):
    '''
    根据机械臂拆除\装箱中对物体的旋转角度和物体放置前后的位移
    :param disassemble: 为真时，表示使用curr_hole_mask计算对应关系；为假时，表示使用curr_object_mask计算对应关系
    :param theta: 旋转角度，逆时针旋转为正，在dissamble=False时，就是obj_mask逆时针旋转，为True时,是hole_mask顺时针旋转
    :param init_point: 物体在洞中的位置
    :param final_point: 物体在外面的位置
    :param init_mask_coord:物体在盒子内的mask的坐标形式
    :param final_mask_coord:物体在盒子外的mask的坐标形式
    :param visual:是否可视化mask
    :return corrs:corrs坐标
    :return corrs_mask:当前时间步生成的corrs可视化图像
    '''
    # init_point = get_curr_point(dirname, init=True)  # 这是洞的中心点
    # final_point = get_curr_point(dirname,init=False)
    #
    # final_pose = np.loadtxt(os.path.join(dirname,'final_pose.txt'))
    # theta = np.arctan2(-final_pose[1, 0], final_pose[0, 0])
    init_point = np.array([init_point[0],init_point[1]])
    final_point = np.array([final_point[0],final_point[1]])

    if disassemble:
        assert init_mask_coord.shape[1] == 2
        rot_mtx = gen_rot_mtx(theta,isdegree=True)
        hole_after_rot = rot_around_point(rot_mtx,init_mask_coord,init_point)
        translation = final_point - init_point
        hole_after_rot = hole_after_rot + translation
        hole_after_rot[:, 0] = np.clip(hole_after_rot[:, 0], 0, cfg.a_h-1)
        hole_after_rot[:, 1] = np.clip(hole_after_rot[:, 1], 0, cfg.a_w-1)
        assert hole_after_rot.shape[1] == 2
        corrs = np.hstack((init_mask_coord,hole_after_rot))

    else:
        assert final_mask_coord.shape[1] == 2
        rot_mtx = gen_rot_mtx(-theta,isdegree=True)
        obj_after_rot = rot_around_point(rot_mtx,final_mask_coord,final_point)
        translation = init_point - final_point
        obj_after_rot = obj_after_rot + translation
        obj_after_rot[:, 0] = np.clip(obj_after_rot[:, 0], 0, cfg.a_h - 1)
        obj_after_rot[:, 1] = np.clip(obj_after_rot[:, 1], 0, cfg.a_w - 1)
        assert obj_after_rot.shape[1] == 2
        corrs = np.hstack((obj_after_rot,final_mask_coord))

    corrs = corrs.astype('int')

    corrs_mask = np.zeros((cfg.a_h, cfg.a_w), dtype=np.uint8)
    corrs_mask[corrs[:, 0], corrs[:, 1]] = 255
    corrs_mask[corrs[:, 2], corrs[:, 3]] = 255
    #可视化
    if visual:
        cv2.imshow('corrs_mask',corrs_mask)
    return corrs, corrs_mask






def gen_pose(dirname,angle,isdegree):
    '''
    生成init_pose.txt和final_pose.txt
    :param dirname:存储文件的位置
    :param angle:逆时针的角度
    :param isdegree:
    :return:
    '''
    #默认初始角度为0度
    init_mtx = np.eye(4)
    init_mtx[2,2] = -1
    init_mtx[1,1] = -1
    np.savetxt(os.path.join(dirname,'init_pose_test.txt'),init_mtx)

    if isdegree:
        angle = math.radians(angle)
    cos_rad = math.cos(angle)
    sin_rad = math.sin(angle)
    rot_mtx = np.eye(4)
    rot_mtx[:2,:2] = [[cos_rad,-sin_rad],[-sin_rad,-cos_rad]]
    rot_mtx[2,2] = -1
    np.savetxt(os.path.join(dirname,'final_pose_test.txt'),rot_mtx)






def erode(mask,kernel_size,iterations):
    #腐蚀运算
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size,kernel_size))
    erode_mask = cv2.erode(mask, kernel, iterations=iterations)
    return erode_mask


def dilate(mask,kernel_size,iterations):
    #膨胀运算
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size,kernel_size))
    dilate_mask = cv2.dilate(mask, kernel, iterations=iterations)
    return dilate_mask


def open_morph(mask,kernel_size,iterations):
    #开运算：先腐蚀后膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    open_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return open_mask


def close_morph(mask,kernel_size,iterations):
    #闭运算：先膨胀后腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    close_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return close_mask


def get_half_centroid_mask(mask, left_half):
    '''
    根绝left_half的真假情况,去掉中心点在右半边或左半边的连通域
    :param mask:二值化mask
    :param left_half:是否要保留左半边的连通域
    :return:
    '''
    w = mask.shape[1] #图片的宽
    w_half = int(w // 2)
    contours = get_exter_contours(mask, 'none')
    for cnt in contours:
        cx,cy = get_centroid(cnt)
        if (left_half and cx > w_half) or (not left_half and cx < w_half):
            cv2.drawContours(mask, [cnt], 0, 0, -1)
    return mask

def get_half_mask(mask,left_half, tole):
    h = mask.shape[0]
    w = mask.shape[1]
    w_half = int(w // 2)
    if left_half :
        mask = np.hstack((mask[:, :w_half+tole], np.zeros((h, w_half-tole))))
    else:
        mask = np.hstack((np.zeros((h, w_half - tole)), mask[:, w_half - tole:]))
    return mask

def color_space_get_all_obj(img,k, color_space_name, range_dict, visual,message):
    '''
    从bgr图片的得到某个颜色空间下分割的obj_mask
    :param img:bgr图片
    :param visual:是否可视化mask
    :param color_space_name: 要分割的颜色空间
    :param range_dict:存放范围的字典
    :param message:
    :return:
    '''
    obj_lower = range_dict['obj_{}_lower'.format(color_space_name)]
    obj_upper = range_dict['obj_{}_upper'.format(color_space_name)]
    name_space_img = convert_image(img, color_space_name)
    mask_obj = cv2.inRange(name_space_img, obj_lower, obj_upper)

    mask_obj = erode(mask_obj, 3, 4)
    mask_obj = remove_small_area(mask_obj,1000,False,'')
    mask_obj = dilate(mask_obj,3, 4)
    mask_obj = cv2.medianBlur(mask_obj, k)
    mask_obj = remove_inner_black(mask_obj,False)
    mask_obj = get_half_centroid_mask(mask_obj, left_half=True)
    if visual:
        cv2.imshow('{} mask_all_obj'.format(message), mask_obj)
    return mask_obj


def mask2coord(mask):
    coord = np.column_stack(np.where(mask))
    return coord


def coord2mask(coord,h,w,visual):
    mask_layer = np.zeros((h,w))
    mask_layer[coord[:, 0],coord[:, 1]] = 1
    if visual:
        cv2.imshow('mask from coord', mask_layer)
    return mask_layer


def is_grayscale(img):
    #判断是否为灰度图
    if img.ndim == 2 or img.shape[2] == 1:
        return True
    else:
        return False


def apply_mask_to_img(mask,imgs,color2gray,visual, mask_info):
    '''
    用mask把img_list中的图像分割出来，其中mask=0的位置全涂黑，否则使用原图像素值
    :param mask: 二维的二值mask
    :param imgs: 所有图片,可以是单张图片或图片列表
    :param color2gray: 是否把彩色图像转为灰度图像
    :param visual: 是否可视结果
    :param mask_info:mask相关信息，用以生成不同的mask窗口
    :return: 分割后的图像或图像列表
    '''
    if isinstance(imgs, list):
        apply_list = []
        for i,img in enumerate(imgs):#enumerate中对list的处理是隔离的，不会影响原来的list
            # print(i)
            if is_grayscale(img):
                img = np.where(mask,img,0)
            else:
                img = np.where(np.repeat(mask[:, :, np.newaxis], 3, axis=-1), img, 0)
                if color2gray:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            apply_list.append(img)
            if visual:
                cv2.imshow('apply {} mask to img: img {} in img list'.format(mask_info, i), img)
        return apply_list
    else:
        if is_grayscale(imgs):
            img = np.where(mask,imgs,0)
        else:
            img = np.where(np.repeat(mask[:, :, np.newaxis], 3, axis=-1), imgs, 0)
            if color2gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if visual:
            cv2.imshow('apply {} mask to img'.format(mask_info), img)
        return img


def put_mask_on_img(mask, imgs, visual, mask_info):
    '''
    把半透明的红色mask覆盖在img_list中的所有图像上并（在visual=True时）显示
    :param mask: 二维的二值mask
    :param imgs: 所有图片,可以是单张图片或图片列表
    :param visual: 是否可视化
    :param mask_info:mask相关信息，用以生成不同的mask窗口
    :return: mask覆盖在图像上的三通道图像或图像列表
    '''
    mask_vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    if isinstance(imgs, list):
        put_list = []
        for i, img in enumerate(imgs):
            if is_grayscale(img):
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            mask_vis[:, :, 2] = np.where(mask, 255, 0)#只对bgr的r通道赋值,也就是给vis涂上红色
            mask_on_img = cv2.addWeighted(img, 0.5, mask_vis, 0.5, 0)
            put_list.append(mask_on_img)
            if visual:
                cv2.imshow('put {} mask on img: img {} in img list'.format(mask_info, i), mask_on_img)
        return put_list
    else:
        if is_grayscale(imgs):
            imgs = cv2.cvtColor(imgs, cv2.COLOR_GRAY2BGR)
        mask_vis[:, :, 2] = np.where(mask, 255, 0)#只对bgr的r通道赋值,也就是给vis涂上红色
        mask_on_img = cv2.addWeighted(imgs, 1, mask_vis, 0.5, 0)
        if visual:
            cv2.imshow('put {} mask on img'.format(mask_info), mask_on_img)
        return mask_on_img

def get_centroid(cnt):
    '''
    获得某个连通域的质心x,y,若有问题返回-1,-1,xy坐标系是opencv坐标系,x水平向右,y竖直向下
    :param cnt:某个连通域的轮廓
    :return:
    '''
    #得到质心
    M = cv2.moments(cnt) # 获取中心点
    if M["m00"] == 0:
        return -1,-1
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx,cy

def get_exter_contours(mask, method = 'none'):
    if method == 'simple' or method == 'SIMPLE':
        contours, hierarch = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:# if method == 'none' or method == 'NONE'
        contours, hierarch = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def get_all_contours(mask, method = 'none'):
    if method == 'simple' or method == 'SIMPLE':
        contours, hierarch = cv2.findContours(mask.astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    else:# if method == 'none' or method == 'NONE'
        contours, hierarch = cv2.findContours(mask.astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours

def get_ccomp_contours(mask, method = 'none', need_parent = True):
    if method == 'simple' or method == 'SIMPLE':
        contours, hierarch = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    else:# if method == 'none' or method == 'NONE'
        contours, hierarch = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if need_parent:
        # 要找到第一个[3]==-1（没有父亲轮廓 = 自己就是父级）的索引，然后根据[0]得到所有相同层级的轮廓
        index = np.array(np.where(hierarch[0,:,3] == -1)[0])
        contours = np.array(contours,dtype=object)[index]
    return contours

def get_tree_contours(mask, method = 'none', need_hierach_level = -1):
    if method == 'simple' or method == 'SIMPLE':
        contours, hierarch = cv2.findContours(mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:# if method == 'none' or method == 'NONE'
        contours, hierarch = cv2.findContours(mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if need_hierach_level != -1:
        # 要找到第一个[3]==n（没有父亲轮廓 = 自己就是父级）的索引，然后根据[0]得到所有相同层级的轮廓
        if need_hierach_level == 0: #要得到最外层的轮廓
            index = np.array(np.where(hierarch[0, :, 3] == -1)[0])
            contours = np.array(contours)[index]
        elif need_hierach_level == -2: #要得到最内层，没有孩子,同时有父亲的轮廓
            condition = np.logical_and(hierarch[0, :, 2] == -1, hierarch[0, :, 3] != -1)
            index = np.array(np.where(condition)).squeeze()
            contours = np.array(contours)[index]
        else: #要得到内层的某层轮廓，一定都是有父轮廓的
            for i in range(len(contours)):
                while hierarch[0, i, 3] == -1:
                    continue
    return contours


def have_four_shape(cor_num_list):
    have_tri = cor_num_list.count("square") == 1
    have_sqr = cor_num_list.count("triangle") == 1
    have_po = cor_num_list.count("pentagon") == 1
    have_cir = cor_num_list.count("circle") == 1
    return have_tri and have_sqr and have_po and have_cir

def compare_corners(image,correct_count,i,color_space_name, color_space_dict):
    # hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # for k in range(1,18,2):
    k = 5
    cor_num = []
    obj_mask = color_space_get_all_obj(image,k, color_space_name, color_space_dict,False,"medium").astype('int8')
    # obj_mask = cv2.medianBlur(np.uint8(obj_mask), 7)
    contours = get_exter_contours(obj_mask, 'NONE')
    show_image = image.copy()
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        # approx = cv2.approxPolyDP(cnt, 0.0375 * peri, True)
        hull = cv2.convexHull(cnt)
        shape = detect_shape_by_approx(hull)
        cor_num.append(shape)
        for coord in hull:
            cv2.circle(show_image, (coord[0][0], coord[0][1]), 3, (0,0,255), -1)
    # cv2.imshow('after_medium_corner', show_image)
    if have_four_shape(cor_num):
        correct_count['{}'.format(k)] = correct_count['{}'.format(k)] + 1
    else:
        obj_mask = cv2.cvtColor(np.uint8(obj_mask),cv2.COLOR_GRAY2BGR)
        pic = np.vstack((obj_mask, show_image))
        cv2.imshow('error.png'.format(i), pic)
        cv2.imwrite(os.path.join(cfg.data_root, 'error', '{}-error{}.png'.format(color_space_name,i)), pic)
        cv2.waitKey()


def detect_shape_by_approx(approx):
    '''
    根据顶点个数判断形状
    :param approx: 顶点列表
    :return:
    '''
    shape = "unidentified"
    # if the shape is a triangle, it will have 3 vertices
    if len(approx) == 3:
        shape = "triangle"
    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio

        # (x, y, w, h) = cv2.boundingRect(approx)
        # ar = w / float(h)

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" ##if ar >= 0.95 and ar <= 1.05 else "rectangle"

    # if the shape is a pentagon, it will have 5 vertices
    elif len(approx) == 5:
        shape = "pentagon"
    else:#更多顶点认为是圆
        shape = "circle"

    # return the name of the shape
    return shape

'''
以下几个函数都是帮助确定颜色参数的工具
'''

def get_range_from_list(range_list):
    range_array = np.array(range_list, dtype=int)
    assert range_array.shape[1] == 3
    upper_value = []
    lower_value = []
    for i in range(3):
        upper_value.append(np.max(range_array[:, i]))
        lower_value.append(np.min(range_array[:, i]))
    upper_value = np.array(upper_value, dtype=int)
    lower_value = np.array(lower_value, dtype=int)
    return [lower_value, upper_value]

def add_point_color(point, image, color_range_dict):
    '''
    point是(y,x)
    :param point:
    :param image:
    :param color_name:
    :return:
    '''
    point = image[point[0],point[1]][np.newaxis,np.newaxis,:] #
    if color_range_dict.has_key('hsv'):
        point = cv2.cvtColor(point, cv2.COLOR_BGR2HSV)
        color_range_dict['hsv'].append(point.ravel())
    if color_range_dict.has_key('xyz'):
        point = cv2.cvtColor(point, cv2.COLOR_BGR2XYZ)
        color_range_dict['xyz'].append(point.ravel())
    if color_range_dict.has_key('ycrcb'):
        point = cv2.cvtColor(point, cv2.COLOR_BGR2YCrCb)
        color_range_dict['ycrcb'].append(point.ravel())
    if color_range_dict.has_key('hls'):
        point = cv2.cvtColor(point, cv2.COLOR_BGR2HLS)
        color_range_dict['hls'].append(point.ravel())
    if color_range_dict.has_key('lab'):
        point = cv2.cvtColor(point, cv2.COLOR_BGR2Lab)
        color_range_dict['lab'].append(point.ravel())
    if color_range_dict.has_key('luv'):
        point = cv2.cvtColor(point, cv2.COLOR_BGR2Luv)
        color_range_dict['luv'].append(point.ravel())
    if color_range_dict.has_key('bgr'):
        color_range_dict['bgr'].append(point.ravel())

def convert_image(image, color_space_name):
    '''根据名字返回不同的颜色空间的图片,不会改变原图
    '''
    if color_space_name == 'bgr':
        return image.copy()
    if color_space_name == 'hsv':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if color_space_name == 'xyz':
        return cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
    if color_space_name == 'ycrcb':
        return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    if color_space_name == 'hls':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    if color_space_name == 'lab':
        return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    if color_space_name == 'luv':
        return cv2.cvtColor(image, cv2.COLOR_BGR2Luv)


def get_pointed_range(event, x, y, flags, param):
    '''
    鼠标事件响应程序：根据鼠标按键更新param[1]，然后把param[0]中像素值在range = [min(param[1]),max(param[1])]的点显示为白色
    鼠标按键更新方式：①左键选择的像素值被加入param[1]
    ②右键选择的像素值从param[1]中被删除，而且为range边界值实现近似删除，如果该像素值与range两端之一的差值在2之间，则会删除端点值
    ③每按下一次中键，去掉当前param[1]中最新加入的值
    :param event: 鼠标事件
    :param x: 鼠标选择的点的x坐标
    :param y: 鼠标选择的点的y坐标
    :param flags:
    :param param: param =[img,pixel]，分别是图像和列表
    :return: 无返回值，根据鼠标事件响应并更新窗口’update_mask‘中的图像后就结束运行
    '''
    # hsv_img = cv2.cvtColor(param[0], cv2.COLOR_BGR2HSV)
    hsv_img = convert_image(param[0],param[2])
    if event == cv2.EVENT_LBUTTONDOWN:
        print('本次按下左键')
        # print('param.type',type(param[0]))
        add_val = hsv_img[y, x]
        # print('add_val', add_val)
        # if add_val not in param[1]:
        param[1].append(add_val)
        # print('目前像素值列表为：', param[1])
        print('成功添加像素值 {}'.format(add_val))
        # else:
        #     print('像素值{}已在，不再添加'.format(add_val))
        range = get_range_from_list(param[1])
        print('目前像素范围为', range, '\n')
        img = mask_range(param[0].copy(), range, param[2])
        cv2.imshow('update_mask', img)
    if event == cv2.EVENT_RBUTTONDOWN:#TODO：是否要改成建立一个排除列表
        print('本次按下右键')
        remove_val = np.array(hsv_img[y, x])
        if remove_val in param[1]:
            param[1].remove(remove_val)
            print('成功删除像素值{}'.format(remove_val))
        else:
            min_val = min(param[1])
            max_val = max(param[1])
            print('像素值{}不在列表内，无法删除'.format(remove_val))
            distance = remove_val - min_val if remove_val > min_val else min_val - remove_val
            if distance <= 2:
                param[1].remove(min_val)
                print('与最小值距离近，删去最小值')
            distance = remove_val - max_val if remove_val > max_val else max_val - remove_val
            if distance <= 2 and len(param[1]) != 0:
                param[1].remove(max_val)
                print('与最大值距离近，删去最大值')
        if len(param[1]) == 0:
            range = []
        else:
            range = get_range_from_list(param[1])
        # print('目前像素值列表为：', param[1])
        print('目前像素范围为', range[0], '\n', range[1])
        img, = mask_range(param[0].copy(), range,param[2])
        cv2.imshow('update_mask', img)

    if event == cv2.EVENT_MBUTTONDOWN:
        if len(param[1]) == 0:
            range = []
        else:
            param[1].pop()
        range = get_range_from_list(param[1])
        # print('目前像素值列表为：', param[1])
        print('目前像素范围为', range[0], '\n', range[1])
        img = mask_range(param[0].copy(), range,param[2])
        cv2.imshow('update_mask', img)


def update_mask(img, pixel,color_space_name):
    '''
    获得mask所需的像素值范围
    :param img:需要选择范围的图像
    :param pixel:选择的像素值列表
    :return:range:选择的像素值范围
    '''
    cv2.namedWindow('update_mask')
    cv2.imshow('update_mask', img)
    param_event = [img, pixel,color_space_name]
    cv2.setMouseCallback('update_mask', get_pointed_range, param_event)
    key = cv2.waitKey()
    if key == ord('q'):
        cv2.destroyWindow('update_mask')
    range = get_range_from_list(pixel)
    return range


def mask_range(img, range,color_space_name):
    '''
    对img中的像素值在range中的点进行mask，mask经过一定的开运算以去除噪声点
    :param img: 需要被mask的图像
    :param range: mask的像素范围
    :param color_space_name:分割方式的颜色空间
    :return: img：已经被mask的图像；
    '''
    if range != []:
        space_name_img = convert_image(img, color_space_name)
        mask_layer = cv2.inRange(space_name_img, range[0], range[1])
        img_mask = put_mask_on_img(mask_layer, img, False, '')
    return img_mask


def gen_transforms(dir, angle, isdegree, index, N):
    '''
    形式就是旋转矩阵，4*4，角度是final和init之间的差值，从盒子里转向在外物体方向
    注意：transforms.npy是（n-1,4,4）
    逆时针为正
    :param dir: dir路径
    :param angle: 旋转角
    :param isdegree: 是否为角度？是，为角度；否，为弧度
    :param index: 当前时间步编号
    :param N: 盒子中积木的个数
    :return:
    '''
    if isdegree:
        angle = math.radians(angle)
    cos_rad = math.cos(angle)
    sin_rad = math.sin(angle)
    trans = np.eye(4)
    trans[:2, :2] = [[cos_rad, sin_rad], [-sin_rad, cos_rad]]
    trans = trans[np.newaxis,:]


    #获得n-1的矩阵，保存到合适位置:
    # ①当放第一个物体的时候，本次transform.npy为空，后面相邻文件夹都保存第一次,
    # ②第二个物体时，读取本文件夹内已有trasnforms.npy，连接第二个transforms，然后保存到下一个相邻文件夹，依此类推
    # ③最后一个物体时，不需要操作
    # 思路可以简化为，第一个物体时，保存空transforms.npy，并且对于所有不是最后一个物体的时间步，读取本文件夹内已有transforms.npy，连接当前时间步的trans然后保存到下一个相邻文件夹中
    i = index % N

    if i == 0:
        if not os.path.exists(os.path.join(dir, str(index + 1))):
            os.makedirs(os.path.join(dir, str(index + 1)))
        np.save(os.path.join(dir,str(index),'transforms.npy'),np.array([]))
        np.save(os.path.join(dir,str(index+1),'transforms.npy'),trans)
    elif i != N-1:
        if not os.path.exists(os.path.join(dir, str(index + 1))):
            os.makedirs(os.path.join(dir, str(index + 1)))
        past_trans = np.load(os.path.join(dir,str(index),'transforms.npy'))
        curr_trans = np.concatenate((past_trans,trans),axis=0)
        np.save(os.path.join(dir, str(index+1), 'transforms.npy'), curr_trans)

def visual_shape(image, cnt, shape):
    '''
    在图像上画出边框和标注结果,会改变传入的参数image
    :param image:
    :param cnt:
    :param shape:
    :return:
    '''
    cx, cy = get_centroid(cnt)
    cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
    cv2.putText(image, shape, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow('shape',image)

def mean_filter(img, k = 10):
    kernel = np.ones((k, k)) / 100
    img = cv2.filter2D(img, -1, kernel)
    return img

# def update_tool(image):
    # pixel_range = update_mask(image, pixel_range)
    # print('======new_range====', pixel_range)

def get_intersection(mask1, mask2,visual, message):
    mask = mask1 & mask2
    if visual:
        cv2.imshow('{}_intersection'.format(message), mask)
    return mask

def get_union(mask1, mask2,visual, message):
    mask = mask1 | mask2
    if visual:
        cv2.imshow('{}_union'.format(message), mask)
    return mask


# def detect_corner(image_1channel,visual,channel_name):
#     '''
#     从单通道图像中检测角点
#     :param image_1channel:
#     :param visual:
#     :param channel_name:
#     :return:
#     '''
#     corners = cv2.goodFeaturesToTrack(image_1channel, 25, 0.01, 10).astype('int64')
#     for i in corners:
#         x, y = i.ravel()
#         cv2.circle(image_1channel, (x, y), 3, (0,255,0), -1)
#     if visual:
#         cv2.imshow('detect_corner_in_{}_channel'.format(channel_name),image_1channel)


def get_avaliable_part(mask, ref_mask, visual):
    #mask中只留下与ref_mask几乎重合的完整连通域
    contours = get_exter_contours(mask,'simple')
    for cnt in contours:
        cx, cy = get_centroid(cnt)
        if ref_mask[cy, cx] == 0:
            cv2.drawContours(mask, [cnt], 0, 0, -1)
    if visual:
        cv2.imshow('inter',mask)
    return mask




def detect_circle_hough(mask,canny_high, regular_th, visual):
    '''
    :param mask: 单通道图像
    :param canny_high:用于canny滤波器的高阈值，低阈值是该值的一半
    :param regular_th:该值越低，可以检测出更多不太规则的圆；值越高，检测到更规则的圆
    :param visual:可视化
    :return:
    '''
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1.5, 200,
                               param1=canny_high, param2=regular_th, minRadius=0, maxRadius=0)
    if circles is None:
        if visual:
            cv2.imshow('circle', mask)
        return circles, mask
    circles = np.uint16(np.around(circles))
    mask_show = mask.copy()
    ##绘制圆
    for i in circles[0]:
        cv2.circle(mask_show, (i[0], i[1]), i[2], 255, 2)
        # cv2.circle(mask_show, (i[0], i[1]), 2, 255, 2)
    if visual:
        cv2.imshow('circle', mask_show)
    return circles, mask_show

def get_convinced_circle(channel_mask):
    reg_th = 100
    # channel_mask = cv2.medianBlur(channel_mask, 5)
    # 如果没检测出来，把regular_th降低1再测，直到检测出一个圆
    while reg_th >= 50:
        circles, mask = detect_circle_hough(channel_mask, 20, reg_th, False)
        reg_th -= 8
        if circles is not None:
            min_radius_index = np.argmin(circles[0][0][2])
            circle = circles[min_radius_index]
            return circle.astype('int')
    return None



def detect_line_houghP(mask, threshod, max_line_gap, visual, message):
    '''
    用cv2.HoughLinesP实现的直线检测
    :param mask: 既可以是二值图像也可以是单通道图像
    :param min_line_len:
    :param max_line_gap:
    :param visual:
    :param message:
    :return:
    '''
    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshod, 0, max_line_gap)
    line_mask = np.uint8(np.zeros(mask.shape))
    if lines is None:
        if visual:
            cv2.imshow('houghP_line_{}'.format(message), line_mask)
        return lines, line_mask
    for line in lines:
        for x1, y1, x2, y2 in line:
            # 给定两点  在原始图片绘制线段
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 1)
    if visual:
        cv2.imshow('houghP_line_{}'.format(message), line_mask)
    return lines, line_mask


def adap_detct_line_hough(mask, visual):
    # 自动检测最长的线段
    lines = cv2.HoughLines(mask, 1, np.pi / 180, adap_th)



def detect_line_hough(mask, thresh, visual):
    '''
    用cv2.HoughLines实现的直线检测
    :param mask:只能是二值mask
    :param thresh:
    :param visual:
    :return:
    '''
    w = mask.shape[1]
    line_mask = np.zeros(mask.shape, dtype=np.uint8)
    # mask_3channel = np.repeat(mask[:, :, np.newaxis], 3, axis=-1)
    lines = cv2.HoughLines(mask, 1, np.pi/180, thresh)
    if lines is None:
        if visual:
            cv2.imshow('hough_line',line_mask)
        return lines, line_mask
    for line in lines:
        draw_a_line(line,line_mask, 255, 1)
    if visual:
        cv2.imshow('hough_line',line_mask)
    return lines, line_mask


def get_convinced_lines(edge_mask,obj_mask, min_len, max_gap):
    # print('get_conviced_line')
    #分别得到hough和houghP的mask
    lines, hough_line_mask = detect_line_hough(edge_mask, min_len, False)
    _, houghP_line_mask = detect_line_houghP(edge_mask, min_len, max_gap, False, 'in get convinced line')
    #求交集
    # convinced = get_intersection(hough_line_mask, houghP_line_mask, False, '')
    if lines is None:
        return lines
    groups = get_lines_group(lines, obj_mask.shape, 40)
    # print('groups', groups)
    proper_lines = []
    for group in groups:
        proper_line = find_proper_line(group, houghP_line_mask)
        proper_lines.append(proper_line)
    return proper_lines

def draw_lines(lines, mask, visual, message):
    #在mask上画多个直线，会改变mask
    for line in lines:
        draw_a_line(line, mask, 255, 1)
    if visual:
        cv2.imshow('lines_{}'.format(message), mask)
    return mask


def get_adjacent_lines(lines, obj_mask):
    '''

    :param lines: 线条的列表，这些线条认为是每个边都有一条
    :param obj_mask:
    :return:
    '''
    lines_num = len(lines)
    adjacent_list = [-1 for i in range(lines_num)]
    for i in range(lines_num):
        for j in range(0, lines_num):
            if could_be_adjecent(i, j, adjacent_list):
                if cross_around_mask(lines[i], lines[j], obj_mask, 30):
                    adjacent_list[i] = j
                    break
    # print('adjacent_list', adjacent_list)

    if adjacent_list.count(-1) == 1 or (adjacent_list.count(-1) == 2 and lines_num == 3):#针对三角形有两个点都有问题
        while adjacent_list.count(-1) != 0:
            for i in range(lines_num):
                if adjacent_list.count(i) == 0 and adjacent_list[(i + lines_num) % lines_num] != adjacent_list.index(-1):
                    adjacent_list[adjacent_list.index(-1)] = i

    # print('adjacent_list', adjacent_list)
    # assert adjacent_list.count(-1) == 0
    # 解决某个交点在图像外，而导致本来应该是邻居的直线没找到邻居
    # if not adjacent_list.count(-1) == 0:
    #     draw_lines(lines, np.zeros(obj_mask.shape), False, 'not adjacent_list.count(-1) <= 1:')
    #     cv2.imshow('accroding to this obj', obj_mask)
    #     cv2.waitKey()

    adjacent_lines = []
    cur = 0
    while len(adjacent_lines) < lines_num:
        adjacent_lines.append(lines[cur])
        cur = adjacent_list[cur]
    # print('type(adjacent_lines)',type(adjacent_lines))
    return adjacent_lines


def could_be_adjecent(i,j,adjacent_list):
    '''
    返回值是bool类型，如果j可以是i的邻居，返回真；若不可以，返回假
    :param i: 要为第i条线找邻居
    :param j: i的邻居的备选项
    :param adjacent_list:邻居列表，list[i]的值是邻居，如果为-1，则还没找到邻居
    :return:
    '''
    if i == j:
        return False
    if adjacent_list[j] == i:
        return False
    if adjacent_list.count(j) != 0:
        return False
    return True
    # (adjacent_list.count(i) != 0 and j == adjacent_list.index(i)) or adjacent_list.count(
    #     j) != 0:  # 是同一条线或别人已经找到它


def get_proper_segment_lines(lines, mask_shape, visual):
    #用数学计算尝试
    coords = []
    assert len(lines) > 1
    for i in np.arange(-1, len(lines) -1):
        j = i + 1
        if np.all(lines[i] == lines[j]):
            raise ValueError('两个相邻线段是一样的')
        coord = get_cross_point(lines[i], lines[j])
        if coord is None:
            mask = np.zeros(mask_shape)
            draw_lines(lines, mask, True, '两个相邻线段没有交点！')
            cv2.waitKey()
            # raise ValueError('两个相邻线段没有交点！')
        coords.append(coord)
    #为这些点画上线段
    segment_mask = np.zeros(mask_shape)
    for i in np.arange(-1, len(coords) - 1):
        cv2.line(segment_mask, coords[i], coords[i+1], 255, 1)
    if visual:
        cv2.imshow('segment_mask',segment_mask)
    return coords, segment_mask


# def move_mask(coords):
#     #当有物体的一个点在外面时
#     if
# def get_bottom_segment(mask):

def get_line_theta_from_2points(two_points):
    '''
    从两个点得到直线方程
    :param two_points: 有两个点的列表，每个点是(x,y)
    :return:弧度角，范围是-pi到pi
    '''
    x0,y0 = two_points[0]
    x1,y1 = two_points[1]
    theta = np.arctan2(y0-y1, x0-x1)
    return theta


# def get_theta_in_reverse_clockwise(clockwise_theta):
#     reverse_clockwise_theta = - ((clockwise_theta + 2 * np.pi) % (2 * np.pi))
#     return reverse_clockwise_theta

def find_near_2points(point_group1, point_group2):
    coord_tole = 15
    two_points = []
    while len(two_points) < 2:
        two_points.clear()
        coord_tole += 1
        for point1 in point_group1:
            for point2 in point_group2:
                if (point1[0] - coord_tole <= point2[0] <= point1[0] + coord_tole) and (
                        point1[1] - coord_tole <= point2[1] <= point1[1] + coord_tole):
                    two_points.append(point2)
    assert len(two_points) == 2
    return two_points

def exist_point_under_bottom(all_coord, bottom_coord):
    for each_coord in all_coord:
        if each_coord[1] > bottom_coord[0][1] + 40:
            return True
    return False

def get_rot_theta(mask, vertex_coord, visual):
    # 根据画出来的直线框出最小矩形，并且计算角度
    # 是物体要逆时针（？）旋转的使得底面朝下的状态
    contours = get_exter_contours(mask, 'simple')
    cnt = contours[0]
    h, w = mask.shape
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    # 坐标变为整数
    box = np.int0(box)
    #比较四个角的坐标和vertex_coord,找到相等的两个点
    center = np.array(get_centroid(cnt))
    if len(vertex_coord) != 4:
        two_points = find_near_2points(box, vertex_coord)
        theta = get_line_theta_from_2points(two_points)
        theta_mtx = gen_rot_mtx_clockwise(theta, False)
        after_coord = np.int0(rot_around_point(theta_mtx, vertex_coord, center))
        bottom_coord = np.int0(rot_around_point(theta_mtx, np.array(two_points), center))
        #检查是否有点在bottom——coord的下面，如果有，则theta+np.pi
        if exist_point_under_bottom(after_coord, bottom_coord):
            theta += np.pi
            theta_mtx = gen_rot_mtx_clockwise(theta, False)
            after_coord = np.int0(rot_around_point(theta_mtx, vertex_coord, center))
    else:
        #正方形 任意一条边都是与最小矩形重合的底边
        theta = get_line_theta_from_2points(vertex_coord[:2])
        theta_mtx = gen_rot_mtx_clockwise(theta, False)
        after_coord = np.int0(rot_around_point(theta_mtx,vertex_coord,center))
    theta = (theta + 2 * np.pi) % (2 * np.pi)
    if visual:
        cv2.fillConvexPoly(mask, after_coord, 255)
        cv2.drawContours(mask, [box], 0, 255, 1)
        cv2.imshow('after rot', mask)
    return theta


def find_proper_line(group, convinced_mask):
    # print('find_proper_line')
    h, w = convinced_mask.shape
    count_list = [0 for k in range(len(group))]
    # print('len(group)',len(group))
    # print('group.shape',group.shape)
    for idx in range(len(group)):
        for rho, theta in group[idx]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            for coordx in np.arange(-w, w):
                row = int(x0 + coordx * (-b))
                col = int(y0 + coordx * a)
                if (0 <= row < h) and (0 <= col < w) and convinced_mask[row, col] == 255:
                    count_list[idx] += 1
    max_index = np.argmax(count_list)
    return group[max_index]

def draw_a_line(line, mask, color, width):
    #会改变原图像
    # print('draw_a_line')
    h, w = mask.shape
    # print(type(line))
    # print(line)
    # print()
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + w * (-b))
        y1 = int(y0 + w * a)
        x2 = int(x0 - w * (-b))
        y2 = int(y0 - w * a)
        mask = cv2.line(mask, (x1, y1), (x2, y2), color, width)
    return mask


def get_lines_group_by_theta(lines):
    # print('---------------------')
    # print('get_lines_group_by_theta')
    # ①按角度分组：从hough的theta入手，排序theta，相差20度之内的所有直线分为一组
    groups = []
    sort_index = np.argsort(lines[..., 1].squeeze())#按theta排序
    assert np.all(lines[..., 1].squeeze() >= 0) and np.all(lines[..., 1].squeeze() < np.pi)
    lines_sorted = lines[sort_index]
    tolerance_radiant = (20 / 180) * np.pi  # 容忍角度在20度内的波动
    group_start_index = 0
    # print('out while')
    while group_start_index < len(lines):
        # print('in while')
        rdt_start = lines_sorted[..., 1][group_start_index]
        condition = np.logical_and(lines_sorted[..., 1].squeeze() >= rdt_start,
                                   lines_sorted[..., 1].squeeze() < rdt_start + tolerance_radiant)
        group_lines_index = np.where(condition)
        new_group = lines_sorted[group_lines_index]
        if new_group.ndim == 2:
            new_group = new_group[np.newaxis,:]
        assert new_group.ndim == 3
        groups.append(new_group)
        group_start_index += len(group_lines_index[0])
    # print('get out while')
    first_group_max_theta = groups[0][-1, 0, 1]
    last_group_min_theta = groups[-1][0, 0, 1]
    # print('going to check')
    if np.pi - last_group_min_theta + first_group_max_theta < tolerance_radiant:
        #最后一组和第一组的直线应当是一组，之前只是因为横跨了pi而被分成两组
        before_len = len(groups)
        groups[0] = np.concatenate((groups[-1], groups[0]))
        groups.pop()
        # print('merge')
        assert len(groups) == before_len - 1
    # print('going to return')
    return groups


def get_lines_group_by_rho(lines, mask_shape,tolerance_rho):
    # print('get_lines_group_by_rho')
    # 按距离分组：排序hough，相差30个像素点之内的所有直线分为一组\
    # tolerance_rho = 40  # 容忍距离波动
    groups = [[] for i in range(2)]
    groups[0].append(lines[0])
    for i in range(1, len(lines)):
        # 检查lines[0]和lines[i]是否有交点，如果图片上没交点且距离大于阈值，认为是两组
        if (np.abs(lines[i][0,0] - lines[0][0,0]) > tolerance_rho) and not cross_in_pic(lines[0], lines[i], mask_shape):
            if groups[1] != []:
                groups[1] = np.concatenate((groups[1], lines[i][np.newaxis, :]))
            else:
                groups[1] = lines[i][np.newaxis, :]
        else:
            groups[0] = np.concatenate((groups[0], lines[i][np.newaxis,:]))
    if groups[1] == []:
        groups.pop();
    # print(groups)
    return groups


def get_lines_group_by_rho_old(lines, obj_mask):
    # print('get_lines_group_by_rho')
    # 按距离分组：排序hough，相差30个像素点之内的所有直线分为一组
    groups = []
    sort_index = np.argsort(lines[..., 0].squeeze()) #按rho绝对值排序
    lines_sorted = lines[sort_index]
    tolerance_rho = 20   # 容忍距离波动
    group_start_index = 0
    while group_start_index < len(lines):
        rho_start = lines_sorted[..., 0][group_start_index]
        condition = np.logical_and(lines_sorted[..., 0].squeeze() >= rho_start,
                                   lines_sorted[..., 0].squeeze() < rho_start + tolerance_rho)
        group_lines_index = np.where(condition)
        new_group = lines_sorted[group_lines_index]
        if new_group.ndim == 2:
            new_group = new_group[np.newaxis, :]
        assert new_group.ndim == 3
        groups.append(new_group)
        group_start_index += len(group_lines_index[0])
        # print('len(group_lines_index[0])',len(group_lines_index[0]))
        # print('group_start_index',group_start_index)
    # 检查任意两组之间的直线是否在画面上有交点，如果有，则这两组合并
    group_num = len(groups)
    # print('rho_group_num',group_num)
    if group_num > 1:
        groups = merge_cross_line_group(groups, obj_mask)
    assert len(groups) <= 2
    return groups

def merge_cross_line_group(groups, obj_mask):
    # groups是按照rho分好组之后的结果
    # 用来解决本应当是一组的直线分成了两组的问题
    # have_merge = False
    for i in range(len(groups) -1):
        if groups[i] is None:
            continue
        for j in range(i + 1, len(groups)):
            if groups[i] is None or groups[j] is None:
                continue
            # print('i = {}, j= {}'.format(i, j))
            line_i = groups[i][0]
            line_j = groups[j][0]
            if cross_in_pic(line_i, line_j, obj_mask):
                # 两条线有交点，这两个线所在的组合为一组
                # print('have_crosss')
                groups[i] = np.concatenate((groups[i], groups[j]))
                groups[j] = None
                # have_merge = True
            # else:
            #     draw_a_line(line_j, obj_mask)
            #     draw_a_line(line_i, obj_mask)
            #     cv2.imshow('two line',obj_mask)
            #     print(line_i, line_j)
    groups = remove_none_in_list(groups)
    return groups# , have_merge


# def merge_until_cannot(groups, obj_mask):
#     have_merge = True
#     while have_merge:
#         groups, have_merge = merge_cross_line_group(groups, obj_mask)
#     return groups

def remove_none_in_list(none_list):
    no_none_index = []
    for i in range(len(none_list)):
        if none_list[i] is not None:
            no_none_index.append(i)
    if len(no_none_index) == 1:
        return [none_list[no_none_index[0]]]
    no_none_list = list(np.array(none_list)[np.array(no_none_index)])
    return no_none_list

def get_cross_point(line1, line2):
    # print('line1',line1)
    # print('line2', line2)
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    # 直线方程为 sin(theta)*y + cos(theta)*x = rho
    a = np.mat([[np.sin(theta1), np.cos(theta1)],
                [np.sin(theta2), np.cos(theta2)]])
    for i in range(a.shape[0]):
        if a[i, i] == 0:
            a[i, i] = 1e-6
    b = np.mat([rho1, rho2]).T
    try:
        y, x = np.linalg.solve(a, b)
    except:
        return None
    return int(x), int(y)

def cross_around_mask(line1, line2,  obj_mask, radius):
    h, w = obj_mask.shape
    #解直线方程或者画图找交点
    coord = get_cross_point(line1, line2)
    if coord is None:
        #无解没交点
        return False
    x, y = coord
    # print('y:', y)
    # print('x', x)
    if (0 <= y < h) and (0 <= x < w) and exist_mask_in_radius(coord, radius, obj_mask):
        return True
    return False


def cross_in_pic(line1, line2,  mask_shape):
    h, w = mask_shape
    #解直线方程或者画图找交点
    coord = get_cross_point(line1, line2)
    if coord is None:
        #无解没交点
        return False
    x, y = coord
    # print('y:', y)
    # print('x', x)
    if (0 <= y < h) and (0 <= x < w):
        return True
    return False


def exist_mask_in_radius(coord, radius, mask):
    mask_except_circle = cv2.circle(mask.copy(), coord, radius, 0, -1)
    # cv2.imshow('mask_except_circle',mask_except_circle)
    return (mask_except_circle != mask).any()

def get_lines_group(lines, mask_shape, tolerance_rho):
    # print('get_lines_group')
    # 在每个角度组中，再按距离分组
    # 分组的等级是平等的
    theta_groups = get_lines_group_by_theta(lines)
    # print('len(theta_groups)',len(theta_groups))
    # print('get out get_lines_group_by_theta')
    all_groups = []
    for theta_group in theta_groups:
        # print('going to rho')
        rho_group = get_lines_group_by_rho(theta_group, mask_shape, tolerance_rho)
        all_groups.extend(rho_group)
    all_groups = remove_none_in_list(all_groups)
    return all_groups

def get_edge_sobel(image, color_name, channel, k = 3, visual = False):
    color_name_image = convert_image(image, color_name)
    onechannel_image = color_name_image[..., channel]
    onechannel_image = cv2.GaussianBlur(onechannel_image, ksize=(3, 3), sigmaX=0, dst=None, sigmaY=None,
                                        borderType=None)
    image_x = cv2.Sobel(onechannel_image, cv2.CV_64F, 1, 0, ksize=k)  # X方向Sobel
    '''
    参数2 depth：必选参数。表示输出图像的深度，-1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度
    参数3和参数4 dx和dy表示的是求导的阶数，0表示这个方向上没有求导，一般为0、1、2
    参数6 ksize：可选参数。用于设置内核大小，即Sobel算子的矩阵大小，值必须是1、3、5、7，默认为3。
    '''
    absX = cv2.convertScaleAbs(image_x)  # 转回uint8
    image_y = cv2.Sobel(onechannel_image, cv2.CV_64F, 0, 1, ksize=k)  # Y方向Sobel
    absY = cv2.convertScaleAbs(image_y)
    # 进行权重融合
    line_mask = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    if visual:
        cv2.imshow('sobel_edge_{}'.format(color_name),line_mask)
    return line_mask

def get_edge_canny(image, color_name, channel, th1, th2, visual):
    color_name_image = convert_image(image, color_name)
    onechannel_image = color_name_image[..., channel]
    onechannel_image = cv2.GaussianBlur(onechannel_image, ksize=(3,3), sigmaX=0, dst=None, sigmaY=None, borderType=None)
    onechannel_image = cv2.medianBlur(onechannel_image, 3)
    color_name_edges = cv2.Canny(onechannel_image, th1, th2)
    if visual:
        cv2.imshow('edges_mask_{}'.format(color_name, channel),color_name_edges)
        # put_mask_on_img(color_name_edges, image, True, 'edges_on_image_{}_()'.format(color_name, channel))
    return color_name_edges

def adap_get_mask_in_color_space(image, color_name, visual):
    lab_image = convert_image(image, 'lab')
    lab_mask = cv2.adaptiveThreshold(lab_image[...,2], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 201, 2)
    lab_mask = remove_small_area(lab_mask, 1000, False, '')
    if color_name == 'hsv':
        color_name_image = convert_image(image, color_name)
        mask = cv2.adaptiveThreshold(color_name_image[...,1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 201, 2)
        lab_mask = get_avaliable_part(mask, lab_mask, False)
    if visual:
        cv2.imshow('{}_mask'.format(color_name), lab_mask)
    return lab_mask.astype('uint8')


def get_each_mask(mask, visual):
    labels = label(mask)
    num = np.max(labels) #背景+白色连通域个数
    each_mask_list = []
    for i in range(1, num+1):
        one_mask = np.where(labels == i, mask, 0)
        each_mask_list.append(one_mask)
    return each_mask_list


def get_obj_mask(image, only_upper_surface):
    # 生成用来去噪的mask
    if only_upper_surface:
        # mask_hsv = color_space_get_all_obj(image, 9, 'hsv', color_space_range_dict, False, 'hsv')
        mask_lab = color_space_get_all_obj(image, 5, 'lab', color_space_range_dict, False, 'lab')
        # mask_hsv = dilate(mask_hsv, 5, 3)
        mask_lab = dilate(mask_lab, 5, 3)
    else:
        # mask_hsv = adap_get_mask_in_color_space(image, 'hsv', False)
        mask_lab = adap_get_mask_in_color_space(image, 'lab', False)
        # mask_hsv = dilate(mask_hsv, 3, 2)
        mask_lab = dilate(mask_lab, 3, 1)
        mask_lab = remove_surrounding_white(mask_lab,False)
    return mask_lab
    # return mask_hsv, mask_lab

def get_convinced_mask(mask_hsv,mask_lab):
    all_mask = get_intersection(mask_hsv, mask_lab, False, 'two_mask')
    all_mask = remove_big_area(all_mask, 30000, False, 'after remove big area')
    all_mask = remove_small_area(all_mask, 1000, False, '')
    all_mask = get_half_centroid_mask(all_mask, True)
    return all_mask.astype('uint8')


# def get_all_edge_mask(image, method, only_upper_surface, visual):
#     if method == 'sobel':
#         hsv_edge = get_edge_sobel(image, 'hsv', 2, 3, False)
#         lab_edge = get_edge_sobel(image, 'lab', 0, 3, False)
#         # 把灰度图二值化
#         hsv_edge = np.uint8(np.where(hsv_edge > 7, 255, 0))
#         lab_edge = np.uint8(np.where(lab_edge > 7, 255, 0))
#     else: #canny
#         hsv_edge = get_edge_canny(image, 'hsv', 2, 20, 30, False)
#         lab_edge = get_edge_canny(image, 'lab', 0, 20, 30, False)
#     # cv2.waitKey()
#     #生成用来去噪的mask
#     all_edge = get_union(hsv_edge, lab_edge, False, 'two_edge')
#     mask_hsv, mask_lab = get_surface_mask(image, only_upper_surface)
#     all_mask = get_union(dilate(mask_hsv,3,1), mask_lab, True, 'two_mask')
#     # hsv_edge = get_intersection(hsv_edge, mask_hsv, False, 'hsv')
#     # lab_edge = get_intersection(lab_edge, mask_lab, False, 'lab')
#     all_edge = get_intersection(all_edge, all_mask, False, 'all')
#     # all_edge = remove_scattered_pix(all_edge, 5, False)
#     all_edge = get_half_mask(all_edge,True, 20)
#
#     if method == 'sobel': #sobel边缘需要精细化
#         hsv_edge = remove_small_area(hsv_edge, 200, False, 'sobel_hsv')
#         lab_edge = remove_small_area(lab_edge, 200, False, 'sobel_lab')
#         # hsv_edge = remove_inner_white(hsv_edge, True, 'sobel_hsv')
#         # lab_edge = remove_inner_white(lab_edge, True, 'sobel_lab')
#     # all_edge = get_union(hsv_edge, lab_edge, False, 'two_edge')
#     # all_edge = close_morph(all_edge, 3, 1)
#     if method == 'sobel':
#         all_edge = erode(all_edge, 3, 2)
#         # all_edge = cv2.medianBlur(all_edge, 3)
#     #此处不能remove_small，也不能remove_inner——white，不然边缘会断
#     # all_edge = remove_scattered_pix(all_edge, 30, False)#对于三角形上表面中间的噪声没用
#     if visual:
#         cv2.imshow('all_edge_{}'.format(method),all_edge)
#     return all_edge, all_mask

def get_all_edge(image, restrict_mask, visual):
    hsv_edge = get_edge_canny(image, 'hsv', 2, 20, 30, False)
    lab_edge = get_edge_canny(image, 'lab', 0, 20, 30, False)
    all_edge = get_union(hsv_edge, lab_edge, False, 'two_edge')

    all_edge = get_intersection(all_edge, restrict_mask, False, 'all')
    # all_edge = remove_scattered_pix(all_edge, 5, False)
    all_edge = get_half_mask(all_edge,True, 20)
    # all_edge = close_morph(all_edge, 3, 1)
    if visual:
        cv2.imshow('all_edge', all_edge)
    return all_edge.astype('uint8')


def draw_label_on_image(image, result_dict, visual):
    for geometry, info in result_dict.items():
        if geometry == 'circle':
            cv2.circle(image, info[0], info[2] , (0,255,0), 2)
        else:
            cv2.drawContours(image, [np.array(info[2])], -1, (0,255,0), 2)
        cv2.putText(image, geometry, info[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    if visual:
        cv2.imshow('result',image)
    return image
# retval, labels_cv, stats, centroids = cv2.connectedComponentsWithStats(img, ltype=cv2.CV_32S)


def get_point2line_distance(points, line):
    # points 是(x, y)
    # line是rho和theta
    return np.abs(np.cos(line[0,1])* points[0] + np.sin(line[0,1]) * points[1] - line[0, 0])


def remove_short_or_twisty(mask, line_th):
    # 原理：去掉霍夫直线检测不出来的部分
    _, existline_mask = detect_line_hough(mask, line_th, False)
    mask = np.where(np.logical_and(mask == 255, existline_mask == 0), 0, mask)
    return mask

def get_all_contour_points(contours):
    all_cnt = contours[0]
    for i in range(1, len(contours)):
        all_cnt = np.concatenate((all_cnt, contours[i]))
    return all_cnt

def get_hull_include_any(mask, visual):
    #返回一个ndarray和其mask，代表了圈出画面内所有内容的凸包
    contours = get_exter_contours(mask, 'none')
    all_cnt = get_all_contour_points(contours)
    hull = cv2.convexHull(all_cnt)
    hull_mask = cv2.drawContours(np.zeros(mask.shape), [hull], -1, 255, 1).astype('uint8')
    if visual:
        cv2.imshow('get_hull_include_any', hull_mask)
    return hull, hull_mask

def get_approxPolyDP_inclue_any(mask, visual):
    contours = get_exter_contours(mask, 'none')
    all_cnt = get_all_contour_points(contours)
    peri = cv2.arcLength(all_cnt, False)
    approx_points = cv2.approxPolyDP(all_cnt, 0.0375 * peri, True)
    if visual:
        approx_mask = cv2.drawContours(np.zeros(mask.shape), [approx_points] , -1, 255, 1).astype('uint8')
        cv2.imshow('approxPolyDP_inclue_any', approx_mask)

def remove_lines_close2ref_line(mask, ref_line, tole_theta1, tole_rho1, tole_theta2, hough_min, method):
    # 用hough去除
    mask_lines, draw_mask = detect_line_hough(mask, hough_min, False)
    draw_a_line(ref_line, draw_mask, 128, 2)
    cv2.imshow('with ref_line', draw_mask)
    assert mask_lines is not None
    # tole_theta1 = 10
    # tole_rho1 = 20
    # tole_theta2 = 20

    def cond1(line, ref_line, tole_theta, tole_rho):
        if abs(line[0, 1] - ref_line[0, 1]) < (tole_theta * np.pi / 180) and abs(line[0, 0] - ref_line[0, 0]) < tole_rho:
            return True
        return False

    def cond2(line, ref_line, tole_theta):
        if abs(line[0, 1] - ref_line[0, 1]) < (tole_theta * np.pi / 180) and cross_in_pic(line, ref_line, mask.shape):
        # if abs(line[0, 1] - ref_line[0, 1]) < (tole_theta * np.pi / 180) and cross_around_mask(line, ref_line, mask, 30):
            return True
        return False

    if method == 0:
        # 处理逻辑,不分组，如果一条线满足要求，就去掉这条线
        for mask_line in mask_lines:
            if cond1(mask_line, ref_line, tole_theta1, tole_rho1) : #or cond2(mask_line, ref_line, tole_theta2):
                draw_a_line(mask_line, mask, 0, 2)
                draw_a_line(mask_line, draw_mask, 0, 2)
                print('mask_line',mask_line,'ref_line',ref_line)
                cv2.imshow('with ref_line', draw_mask)
                cv2.waitKey()
    elif method == 1:
        # 处理逻辑:如果不能全组满足，就全组跳过
        lines_groups = get_lines_group(mask_lines, mask.shape, tole_rho1)
        for lines in lines_groups:
            jump_flag = False
            for i in range(len(lines)):
                if not (cond1(lines[i], ref_line, tole_theta1, tole_rho1) or cond2(lines[i], ref_line, tole_theta2)):
                    #如果不能全组满足，就全组跳过
                    jump_flag = True
                    break
            if jump_flag == False:
                for line in lines:
                    draw_a_line(line, mask, 0, 2)
    else:
        # 处理逻辑:如果组内有任何一个线满足要求，就整组消除
        hull_groups = get_lines_group(mask_lines, mask.shape, tole_rho1)
        for lines in hull_groups:
            for i in range(len(lines)):
                # if not (abs(lines[i][0, 1] - ref_line[0, 1]) > (tole_theta * np.pi / 180) or abs(
                #         lines[i][0, 0] - ref_line[0, 0]) > tole_rho or cross_in_pic(lines[i], ref_line, mask.shape)):
                if cond1(lines[i], ref_line, tole_theta1, tole_rho1) or cond2(lines[i], ref_line, tole_theta2):
                    # 如果组内有任何一个线满足要求，就整组消除
                    for line in lines:
                        draw_a_line(line, mask, 0, 2)
                    break
    return mask


def remain_min_or_max_line(lines, center, remain_max):
    #保留这一组直线中，距离mask凸包的最远或最近的一些直线
    #会改变mask
    distance = []
    for i in range(len(lines)):
        distance.append(get_point2line_distance(center, lines[i]))
    distance = np.array(distance)
    if remain_max:
        m_distance = np.max(distance)
        m_distance_index = np.argmax(distance)
    else:
        m_distance = np.min(distance)
        m_distance_index = np.argmin(distance)
    remove_index = np.where(abs(distance - m_distance) > 8)
    # print('remove_index', remove_index)
    ref_line = lines[m_distance_index]
    if len(remove_index) > 0:
        remove_lines = lines[remove_index[0]]  # 确定要去掉的线
    return remove_lines, ref_line

def get_theta(point1,point2):
    return np.arctan2(point2[0,1] - point1[0,1] , point2[0,0] - point1[0,0])

def remove_twisty_end(mask, visual):
    # 找到mask上所有外轮廓
    # 对于每个轮廓，去除直线末端的短弯曲
    new_mask = np.zeros(mask.shape)
    #每次取两个间隔为3的点，和下一批间隔为3的点，如果他们分别表示的线段角度大于10，则去掉第二批的第二个点
    contours = get_exter_contours(mask,'simple')
    # print('----------------')
    len_poin = 3
    for cnt in contours:
        # print(cnt)
        remain_list = []
        if len(cnt) < 2*len_poin:
            cv2.drawContours(new_mask, [cnt], -1, 255, 1).astype('uint8')
            continue
        for i in np.arange(-len_poin * 2, len(cnt) - len_poin * 2):
            a_s = cnt[i]
            a_e = cnt[i+len_poin] #同时是b_a
            b_e = cnt[i+ 2 * len_poin]
            a_theta = get_theta(a_s, a_e)
            b_theta = get_theta(a_e, b_e)
            if (abs(a_theta - b_theta) < 10):
                remain_list.append(i + len_poin)
        cnt = cnt[np.array(remain_list)]
        cv2.drawContours(new_mask, [cnt], -1, 255, 1).astype('uint8')
    if visual:
        cv2.imshow('after remove twisty end', new_mask)
    return new_mask.astype('uint8')



def remove_underside_line(mask, visual):
    #分别处理①最外层的凸包轮廓（包含底面的边缘）和②内部的上表面边缘和中间的噪声
    # 上表面边缘A的特性：最靠近外部凸包的唯一一条直线
    # 对于①：去除凸包轮廓上，最靠近A的那条线段
    # 对于②：去除中间的噪声
    #输入是只有一个图像的边缘的mask

    # 找到外边缘
    hull, hulls_mask = get_hull_include_any(mask, False)

    center = get_centroid(hull)
    # cv2.imshow('hulls_mask', hulls_mask)
    # hull_lines, _ = detect_line_hough(hulls_mask, 15, True)
    # cv2.waitKey()

    #如果凸包中有其他线段，则可能是上表面的边缘，也可能是中间的噪声
    except_hulls = cv2.drawContours(mask, [hull], -1, 0, 4).astype('uint8')
    except_hulls = remove_scattered_pix(except_hulls, 5, False)
    # cv2.imshow('except_hulls',except_hulls)
    if (except_hulls == 0).all():
        return hulls_mask
    lines, existline_mask = detect_line_hough(except_hulls, 20, False)
    # cv2.imshow('exist line_mask', existline_mask)
    # cv2.waitKey()
    if lines is None or len(lines) == 1:# 凸包中没有其他线段,所以整个凸包是有用的
        return hulls_mask
    # 去掉没检出直线的部分，认为是噪音
    except_hulls = np.where(np.logical_and(except_hulls == 255, existline_mask == 0), 0, except_hulls)
    # cv2.waitKey()
    groups = get_lines_group(lines, mask.shape, 20)

    for lines in groups:
        #每个组处理，用来处理多个侧面边缘和多个底面边缘露出的情况
        # 中间存在线,保留距离凸包中心最远的那条线
        remove_lines, line_a = remain_min_or_max_line(lines, center, False)
        for i in range(len(remove_lines)):
            except_hulls = draw_a_line(remove_lines[i], except_hulls, 0, 1)

        # except_hulls = dilate(except_hulls, 3, 1)
        # cv2.imshow('except_hulls_for_a_theta_line', except_hulls)
        # cv2.waitKey()

        # 增强A，为此要找到线段端点并在except_hulls上画出
        # # assert (except_hulls != 0).any()
        # _,_, except_hulls = find_hough_end_point(line_a, except_hulls, False)


        # 再去除凸包中靠近A的那个线段,特点是与保留线的theta相近,且rho距离小于20
        cv2.imshow('before do sth at hulls_mask', hulls_mask)
        print('before this')
        hulls_mask = remove_lines_close2ref_line(hulls_mask, line_a, 15, 15, 20, 15, 0)
        print('ai this')
        cv2.imshow('after do sth at hulls_maske', hulls_mask)
        # cv2.waitKey()
    # 增强要保留的线
    useful_edge = get_union(except_hulls, hulls_mask, False, 'union')
    print('goint to out')
    #此时应该没有其他的平行线段，如果有，还要去掉最外延的线段，侧面的短直线能否也一起处理掉

    # mask_hsv, mask_lab = get_surface_mask(image, False)
    # hsv_edge = get_intersection(useful_edge, mask_hsv, False, 'hsv')
    # lab_edge = get_intersection(useful_edge, mask_lab, False, 'lab')
    # useful_edge = get_union(hsv_edge, lab_edge,True, 'two_edge')
    # useful_edge = dilate(useful_edge,3,1)
    # useful_edge = cv2.medianBlur(useful_edge, 5)
    # useful_edge = remove_short_or_twisty(useful_edge, 5)
    # useful_edge = remove_scattered_pix(useful_edge, 10, False)
    # get_approxPolyDP_inclue_any(useful_edge, True)
    # useful_edge = remove_twisty_end(useful_edge, False)
    # approx = cv2.approxPolyDP(cnt, 0.0375 * peri, True)
    # _, useful_edge = get_hull_include_any(useful_edge, False)
    #由于存在侧面的短直线，不能返回凸包
    # useful_edge = dilate(useful_edge,3,1)
    # detect_line_hough(useful_edge, 40, False)
    # detect_line_houghP(useful_edge, 40, 10, True ,' ')
    # useful_edge = cv2.medianBlur(useful_edge,3)
    if visual:
        cv2.imshow('after remove_underside_line', useful_edge)
    return useful_edge

def find_hough_end_point(line, mask, visual):
    #找到一个霍夫变换的直线的端点
    line_mask = draw_a_line(line, np.zeros(mask.shape), 255, 2)
    # cv2.imshow('max_line_mask', max_line_mask)
    xand_coords = np.where(np.logical_and(line_mask == mask, line_mask != 0))#找到的是相等点的所有点坐标
    # print('xand',xand_coords)
    if xand_coords[0] != [] and xand_coords[1] != []:#均非空
        min_row = np.argmin(xand_coords[0])
        max_row = np.argmax(xand_coords[0])
        one_endpoint = (xand_coords[1][min_row],xand_coords[0][min_row])
        another_endpoint = (xand_coords[1][max_row],xand_coords[0][max_row])
    hough_segment_mask = cv2.line(np.zeros(mask.shape), one_endpoint, another_endpoint, 255, 1)
    if visual:
        cv2.imshow('find_hough_end_point',hough_segment_mask)
    return one_endpoint, another_endpoint, hough_segment_mask

def remove_hough_line_without_damage():
    # 筛掉间距过大的其他点
    pass


def remove_underside_line_old(mask, visual):
    #分别处理①最外层的凸包轮廓（包含底面的边缘）和②内部的上表面边缘和中间的噪声
    # 上表面边缘A的特性：最靠近外部凸包的唯一一条直线
    # 对于①：去除凸包轮廓上，最靠近A的那条线段
    # 对于②：去除中间的噪声
    #输入是只有一个图像的边缘的mask
    #找到外边缘
    contours = get_exter_contours(mask, 'none')
    hulls = []
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        hulls.append(hull)
    hulls_mask = cv2.drawContours(np.zeros(mask.shape), hulls, -1, 255, 1).astype('uint8')
    # hulls中的每个元素是ndarray （N，1,2）
    #如果凸包中有其他线段，则可能是上表面的边缘，也可能是中间的噪声
    except_hulls = cv2.drawContours(mask, hulls, -1, 0, 4).astype('uint8')
    #
    center = get_centroid(hull)
    lines, _ = detect_line_hough(except_hulls, 40, False)
    distance = []
    if lines is not None and len(lines) > 1: #中间存在线,保留距离凸包中心最远的那条线
        for i in range(len(lines)):
            distance.append(get_point2line_distance(center, lines[i]))
        max_index = distance.index(max(distance))
        #去除非最大值的直线
        for i in range(len(lines)):
            if i == max_index:
                continue
            else:
                except_hulls = draw_a_line(lines[i], except_hulls, 0, 1)
        cv2.imshow('except_hulls', except_hulls)
        # 增强上表面边缘所在的直线，为此要找到线段端点并画出
        max_line_mask = draw_a_line(lines[max_index], np.zeros(mask.shape), 255, 2)
        # cv2.imshow('max_line_mask', max_line_mask)
        xand_coords = np.where(np.logical_and(max_line_mask == except_hulls, max_line_mask != 0))#找到的是相等点的所有点坐标
        min_row = np.argmin(xand_coords[0])
        max_row = np.argmax(xand_coords[0])
        one_endpoint = (xand_coords[1][min_row],xand_coords[0][min_row])
        another_endpoint = (xand_coords[1][max_row],xand_coords[0][max_row])
        cv2.line(except_hulls, one_endpoint, another_endpoint, 255, 1)


        # 再去除凸包中靠近保留线的那个线段,特点是与保留线的theta相近
        hull_lines, _ = detect_line_hough(hulls_mask, 20, False)
        tole_theta = 10
        for hull_line in hull_lines:
            if abs(hull_line[0, 1] - lines[max_index][0, 1]) < tole_theta * np.pi / 180:
                draw_a_line(hull_line, hulls_mask, 0, 1)
                # cv2.imshow('remain', hulls_mask)
                # cv2.imshow('except_hulls2',except_hulls)
        cv2.imshow('hulls_mask', hulls_mask)
        # 增强要保留的线
        useful_edge = get_union(except_hulls, hulls_mask, False, 'union')
        useful_edge = remove_scattered_pix(useful_edge, 3, False)
        # useful_edge = dilate(useful_edge,3,1)
        # detect_line_hough(useful_edge, 40, False)
        # detect_line_houghP(useful_edge, 40, 10, True ,' ')
        # useful_edge = cv2.medianBlur(useful_edge,3)
        if visual:
            cv2.imshow('after remove_underside_line', useful_edge)
        return useful_edge
    return mask


def get_skeleteton(mask, iter_num):
    array = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, \
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, \
             1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]

    def Three_element_add(array):
        array0 = array[:].flatten()
        array1 = np.append(array[1:], np.array([0,0,0]))
        array2 = np.append(array[2:], np.array([[0,0,0],[0,0,0]]))
        arr_sum = array0 + array1 + array2
        return arr_sum[:-2]

    def VThin(image, array):
        NEXT = 1
        height, width = image.shape[:2]
        for i in range(1, height):
            M_all = Three_element_add(image[i])
            for j in range(1, width):
                if NEXT == 0:
                    NEXT = 1
                else:
                    M = M_all[j - 1] if j < width - 1 else 1
                    if (image[i, j] == 0).all() and M != 0:
                        a = np.zeros(9)
                        if height - 1 > i and width - 1 > j:
                            kernel = image[i - 1:i + 2, j - 1:j + 2]
                            a = np.where(kernel == 255, 1, 0)
                            a = a.reshape(1, -1)[0]
                        NUM = np.array([1, 2, 4, 8, 0, 16, 32, 64, 128])
                        sumArr = np.sum(a * NUM)
                        image[i, j] = array[sumArr] * 255
                        if array[sumArr] == 1:
                            NEXT = 0
        return image

    def HThin(image, array):
        height, width = image.shape[:2]
        NEXT = 1
        for j in range(1, width):
            M_all = Three_element_add(image[:, j])
            for i in range(1, height):
                if NEXT == 0:
                    NEXT = 1
                else:
                    M = M_all[i - 1] if i < height - 1 else 1
                    if (image[i, j] == 0).all( ) and M != 0:
                        a = np.zeros(9)
                        if height - 1 > i and width - 1 > j:
                            kernel = image[i - 1:i + 2, j - 1:j + 2]
                            a = np.where(kernel == 255, 1, 0)
                            a = a.reshape(1, -1)[0]
                        NUM = np.array([1, 2, 4, 8, 0, 16, 32, 64, 128])
                        sumArr = np.sum(a * NUM)
                        image[i, j] = array[sumArr] * 255
                        if array[sumArr] == 1:
                            NEXT = 0
        return image

    binary_image = mask.astype('uint8')
    new_image = cv2.copyMakeBorder(binary_image, 1, 0, 1, 0, cv2.BORDER_CONSTANT, value=0)
    for i in range(iter_num):
        VThin(image, array)
        HThin(image, array)
    return image


def get_center(image):
    # mask_hsv, mask_lab = get_obj_mask(image, False)
    mask_lab = get_obj_mask(image, False)
    # all_mask = get_convinced_mask(mask_hsv, mask_lab)
    cv2.imwrite("mask_lab.jpg", mask_lab)
    each_mask_list = get_each_mask(mask_lab, False)

    for each_mask in each_mask_list:
        contours = get_exter_contours(each_mask, 'simple')
        cnt = contours[0]
        center = get_centroid(cnt)
        assert center is not None
        '''
        得到的是cx和cy
        '''
    return center






if __name__ == "__main__":

    top_dir = os.path.join(cfg.data_root, cfg.data_type)
    img_dir = os.path.join(cfg.data_root,'img')
    # points_dit = os.path.join(cfg.data_root,'points')
    rgb_image_num = len(os.listdir(img_dir))//2
    pixel_range = []
    dict = {}
    need_update= False
    select_param = False
    # color_space_range_dict = {}
    color_space_range_dict = {'obj_hsv_lower': np.array([101, 93, 90]), 'obj_hsv_upper': np.array([106, 255, 142]),
                              'obj_ycrcb_lower': np.array([40, 87, 137]), 'obj_ycrcb_upper': np.array([103, 115, 164]),
                              'obj_xyz_lower': np.array([35, 43, 93]), 'obj_xyz_upper': np.array([104, 113, 150]),
                              'obj_hls_lower': np.array([101, 46, 65]), 'obj_hls_upper': np.array([107, 116, 255]),
                              'obj_lab_lower': np.array([53, 121, 94]), 'obj_lab_upper': np.array([113, 132, 108]),
                              'obj_luv_lower': np.array([52, 81, 93]), 'obj_luv_upper': np.array([113, 88, 111])
                              }
    for i in range(1, 18, 2):
        dict['{}'.format(i)] = 0

    color_space_name = 'hsv'
    for i in range(79, rgb_image_num, 8):#range(7,rgb_image_num,8)
        if not os.path.exists(os.path.join(img_dir, 'color{}.png'.format(i))):
            continue
        else:

            image = cv2.imread(os.path.join(img_dir, 'color{}.png'.format(i)))

            w, h, c = image.shape
            half_w = h//2

            #hsv的1通道经过adaptive之后可以很好的分割出整个物体，不要inv，但是还需要各种后处理
            #hsv的2通道在canny下可以得到不错的边缘，但是自适应分割时完全看不出

            #lab的2通道在adaptive分割之后可以分割出整个物体，要inv，保留左半边并去除小区域即可
            #lab 0 通道可用于canny得到两个表面的边缘
            print('=============={}==================='.format(i))
            # mask_0 = cv2.adaptiveThreshold(half_image[...,0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 201, 2)
            mask_hsv,mask_lab = get_obj_mask(image, False)
            all_mask = get_convinced_mask(mask_hsv,mask_lab)
            each_mask_list = get_each_mask(all_mask, False)

            for each_mask in each_mask_list:
                cnt = get_exter_contours(each_mask, 'simple')
                center = get_centroid(cnt)




