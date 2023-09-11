import cv2
import numpy as np
import math
import os
import form2fit.config as cfg
from skimage.measure import label


def remove_small_area(mask, area_th,visual, message):
    '''
        去除小面积连通域
        :param mask: 待处理的mask
        :param area_th: 小于area_th的面积会被消除
        :return: 去除了小面积连通域的mask
    '''
    contours = get_exter_contours(mask, 'none')
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
    contours = get_exter_contours(mask, 'none')
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
    生成顺时针旋转指定角的旋转矩阵
    :param angle:要顺时针旋转的角
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


def get_half_centroid_mask(mask, left_half, tole):
    '''
    根绝left_half的真假情况,去掉中心点在右半边或左半边的连通域,
    :param mask:二值化mask
    :param left_half:是否要保留左半边的连通域
    :param tole:对额外偏移的容忍
    :return:
    '''
    w = mask.shape[1] #图片的宽
    w_half = int(w // 2)
    contours = get_exter_contours(mask, 'none')
    for cnt in contours:
        cx,cy = get_centroid(cnt)
        if (left_half and cx > w_half + tole) or (not left_half and cx < w_half - tole):
            cv2.drawContours(mask, [cnt], 0, 0, -1)
    return mask

def get_half_mask(mask,left_half, tole):
    h = mask.shape[0]
    w = mask.shape[1]
    w_half = int(w // 2)
    if left_half:
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
    mask_obj = get_half_centroid_mask(mask_obj, left_half=True, tole=50)
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



def detect_shape_by_line_count(lines):
    '''
    根据顶点个数判断形状
    :param approx: 顶点列表
    :return:
    '''
    shape = "unidentified"
    # if the shape is a triangle, it will have 3 vertices
    if len(lines) == 3:
        shape = "triangle"
    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(lines) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio

        # (x, y, w, h) = cv2.boundingRect(approx)
        # ar = w / float(h)

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" ##if ar >= 0.95 and ar <= 1.05 else "rectangle"
    elif len(lines) == 5:
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
                               param1=canny_high, param2=regular_th, minRadius=10, maxRadius=45)
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
        circles, mask = detect_circle_hough(channel_mask, 30, reg_th, False)
        reg_th -= 8
        if circles is not None:
            min_radius_index = np.argmin(circles[0][0][2])
            circle = circles[min_radius_index]
            return circle.astype('int')
    return None


#
# def detect_line_houghP(mask, threshod, max_line_gap, visual, message):
#     '''
#     用cv2.HoughLinesP实现的直线检测
#     :param mask: 既可以是二值图像也可以是单通道图像
#     :param min_line_len:
#     :param max_line_gap:
#     :param visual:
#     :param message:
#     :return:
#     '''
#     lines = cv2.HoughLinesP(mask, 1, np.pi *2 / 180, threshod, 0,max_line_gap)
#     line_mask = np.uint8(np.zeros(mask.shape))
#     if lines is None:
#         if visual:
#             cv2.imshow('houghP_line_{}'.format(message), line_mask)
#         return lines, line_mask
#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             # 给定两点  在原始图片绘制线段
#             cv2.line(line_mask, (x1, y1), (x2, y2), 255, 1)
#     if visual:
#         cv2.imshow('houghP_line_{}'.format(message), line_mask)
#     return lines, line_mask


def adap_detct_max_line_hough(mask, min_line_len, visual):
    # 自动检测全局最长的线段
    adap_th = min_line_len
    lines = cv2.HoughLines(mask, 1, np.pi / 180, adap_th)
    if lines is None:
        return None, None
    while lines is not None:
            adap_th += 4
            lines = cv2.HoughLines(mask, 1, np.pi / 180, adap_th)
    if lines is None:
        while lines is None and adap_th > min_line_len:
            adap_th -= 1
            lines = cv2.HoughLines(mask, 1, np.pi / 180, adap_th)
        # 现在lines的数量是可以检测出来的符合条件的最长线段的个数，可能为1,2,3...
        if lines is None:
            return None, None
    if visual:
        line_mask = mask.copy()
        line_mask = draw_lines(lines, line_mask, 255, 1, True, 'hough detected Max len line')
    return lines, np.repeat(np.array([adap_th]), len(lines)) #同时返回这些线的长度

def adap_detect_all_line_hough(mask, min_line_len, visual):
    # 自动分组检测每组最长线段
    # 检测出最长的线段后去掉该线段，去噪点继续检测，直到无线可检
    detect_mask = mask.copy()

    max_lines, max_len = adap_detct_max_line_hough(detect_mask, min_line_len, False)
    all_max_lines = max_lines
    all_max_len = max_len
    if max_lines is None: # 找不到任何直线
        return None,None
    #找到所有方向的最长线段
    while True:
        draw_lines(max_lines, detect_mask, 0, 2, False, 'after remove one mask line')
        max_lines, max_len = adap_detct_max_line_hough(detect_mask, min_line_len, False)
        if max_lines is None:
            break
        else:
            all_max_lines = np.concatenate((all_max_lines, max_lines))
            all_max_len = np.concatenate((all_max_len, max_len))
            assert len(all_max_len) == len(all_max_lines)

    if visual:
        line_mask = np.zeros(mask.shape)
        line_mask = draw_lines(all_max_lines, line_mask, 255, 1, True, ' adap_detect_all_line_hough')
    return all_max_lines, all_max_len


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


# def get_convinced_lines(edge_mask,obj_mask, min_len, max_gap):
#     # print('get_conviced_line')
#     #分别得到hough和houghP的mask
#     lines, hough_line_mask = detect_line_hough(edge_mask, min_len, False)
#     _, houghP_line_mask = detect_line_houghP(edge_mask, min_len, max_gap, False, 'in get convinced line')
#     #求交集
#     # convinced = get_intersection(hough_line_mask, houghP_line_mask, False, '')
#     if lines is None:
#         return lines
#     groups = get_lines_group(lines, obj_mask.shape, 40, False)
#     # print('groups', groups)
#     proper_lines = []
#     for group in groups:
#         proper_line = find_proper_line(group, houghP_line_mask)
#         proper_lines.append(proper_line)
#     return proper_lines

def draw_lines(lines, mask, color, width, visual, message):
    #在mask上画多个直线，会改变mask
    for line in lines:
        draw_a_line(line, mask, color, width)
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

    if adjacent_list.count(-1) == 1 or (adjacent_list.count(-1) == 2 and lines_num == 3):#针对三角形有两个点都有问题
        while adjacent_list.count(-1) != 0:
            for i in range(lines_num):
                if adjacent_list.count(i) == 0 and adjacent_list[(i + lines_num) % lines_num] != adjacent_list.index(-1):
                    adjacent_list[adjacent_list.index(-1)] = i

    # print('adjacent_list', adjacent_list)
    # assert adjacent_list.count(-1) == 0
    # 解决某个交点在图像外，而导致本来应该是邻居的直线没找到邻居
    # if not adjacent_list.count(-1) == 0:
    #     draw_lines(lines, np.zeros(obj_mask.shape), 255, 1,False, 'not adjacent_list.count(-1) <= 1:')
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


def get_proper_segment_lines(lines, mask_shape, line_color, line_width, visual):
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
            draw_lines(lines, mask, 255, 1, True, '两个相邻线段没有交点！')
            cv2.waitKey()
            # raise ValueError('两个相邻线段没有交点！')
        coords.append(coord)
    #为这些点画上线段
    segment_mask = np.zeros(mask_shape)
    coords = np.array(coords)
    cv2.drawContours(segment_mask, [coords], 0, line_color, line_width)
    # for i in np.arange(-1, len(coords) - 1):
    #     cv2.line(segment_mask, coords[i], coords[i+1], 255, 1)
    assert len(coords) == len(lines)
    if visual:
        cv2.imshow('segment_mask',segment_mask)
    return coords, segment_mask


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

#
# def get_line_alpha_b(pt1, pt2):
#     alpha = np.arctan2((pt2[0,1] - pt1[0,1]) , pt2[0,0] - pt1[0,0])
#     # if alpha > np.pi:
#     #     alpha = alpha - 2 * np.pi
#     # 直线方程为
#     b = pt2[0,1] - np.tan(alpha) * pt2[0,0]
#     return alpha, b
#
# def get_rot_theta_new(vertex_coord, center, mask_shape, ori_mask, visual):
#     # 根据任意一条直线计算将这条直线旋转到水平且处于最顶端位置的方法
#     pt1 = vertex_coord[0][np.newaxis, :]
#     pt2 = vertex_coord[1][np.newaxis, :]
#     alpha, b = get_line_alpha_b(pt1, pt2)
#     print('alpha = ',alpha * 180 / np.pi)
#     print('b = ', b)
#     center = np.array([center[0], center[1]])
#     # 直线方程是y = tan(alpha) * x + b
#     y_xc = np.tan(alpha) * center[0] + b
#     if y_xc > center[1]:
#         # 说明中心点在线的上方(从人的视角看的上下)(y轴竖直向下为正方向)
#         print('在上方')
#         rot_theta = alpha
#     else:
#         print('在下方')
#         rot_theta = np.pi + alpha
#     if visual:
#         # 逆时针旋转rot_theta
#         theta_mtx = gen_rot_mtx(rot_theta, False)  #
#         after_coord = np.int0(rot_around_point(theta_mtx, vertex_coord, center))
#         mask = np.zeros(mask_shape).astype('uint8')
#         cv2.line(mask, after_coord[0], after_coord[1], 128, 5)
#         cv2.fillConvexPoly(mask, after_coord, 255)
#         # cv2.imshow('after rot', mask)
#         ori_mask = ori_mask.astype('uint8')
#         cv2.line(ori_mask, vertex_coord[0], vertex_coord[1], 128, 5)
#         cv2.imshow('before rot', ori_mask)
#     return rot_theta

def get_rot_theta(mask, vertex_coord, visual):
    # 根据画出来的直线框出最小矩形，并且计算角度
    # 是物体要逆时针旋转的使得底面朝上的状态
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
        theta = (-1) * get_line_theta_from_2points(two_points)
        theta_mtx = gen_rot_mtx(theta, False) #
        after_coord = np.int0(rot_around_point(theta_mtx, vertex_coord, center))
        bottom_coord = np.int0(rot_around_point(theta_mtx, np.array(two_points), center))
        #检查是否有点在bottom_coord的下面，如果无，则theta+np.pi
        if not exist_point_under_bottom(after_coord, bottom_coord):
            theta += np.pi
            theta_mtx = gen_rot_mtx(theta, False)
            after_coord = np.int0(rot_around_point(theta_mtx, vertex_coord, center))
    else:
        #正方形 任意一条边都是与最小矩形重合的底边
        theta = (-1) * get_line_theta_from_2points(vertex_coord[:2])
        theta_mtx = gen_rot_mtx(theta, False)
        after_coord = np.int0(rot_around_point(theta_mtx,vertex_coord,center))
    theta = (theta + 2 * np.pi) % (2 * np.pi)
    theta = (theta + 2 * np.pi) % (2 * np.pi / len(vertex_coord))
    if visual:
        theta_mtx = gen_rot_mtx(theta, False)  #
        after_coord = np.int0(rot_around_point(theta_mtx, vertex_coord, center))
        cv2.fillConvexPoly(mask, after_coord, 255)
        cv2.drawContours(mask, [box], 0, 255, 1)
        cv2.imshow('after rot', mask)
        # cv2.imwrite(os.path.join(cfg.data_root,'show','{}_after_rot.png'.format(len(vertex_coords))),mask)
    return theta


def find_proper_line(group, convinced_mask):
    h, w = convinced_mask.shape
    count_list = [0 for k in range(len(group))]
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
    h, w = mask.shape
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


def get_lines_group_by_theta(lines,tole_ang):
    # print('---------------------')
    # print('get_lines_group_by_theta')
    # ①按角度分组：从hough的theta入手，排序theta，相差20度之内的所有直线分为一组
    groups = []
    sort_index = np.argsort(lines[..., 1].squeeze())#按theta排序
    assert np.all(lines[..., 1].squeeze() >= 0) and np.all(lines[..., 1].squeeze() < np.pi)
    lines_sorted = lines[sort_index]
    tolerance_radiant = (tole_ang / 180) * np.pi  # 容忍角度在20度内的波动
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
    def lines_are_far(line1, line2, tolerance_rho):
        if np.abs(line1[0,1] - line2[0,1]) > (60 * np.pi/ 180) and ((line1[0,1] < 10 * np.pi / 180) or (np.pi - line1[0,1] < 10 * np.pi / 180)):
            # Done: 当线段几近垂直而分跨pi两侧时, rho的绝对值之差较大
            if np.abs(np.abs(line1[0,0]) - np.abs(line2[0,0])) > tolerance_rho:
                return True
        elif np.abs(line1[0,1] - line2[0,1]) < (10 * np.pi/ 180) and (line1[0,0] * line2[0,0] < 0):
        # Done: 当线段倾斜于原点,正好同角度分跨原点两侧时, rho要比较原值
            if np.abs(line1[0,0] - line2[0,0]) > tolerance_rho:
                return True
        else:
            if np.abs(line1[0,0] - line2[0,0]) > tolerance_rho:
                return True
        return False
    groups = [[] for i in range(2)]
    groups[0].extend(lines[0][np.newaxis, :])
    for i in range(1, len(lines)):
        # 检查lines[0]和lines[i]是否有交点，如果图片上没交点且距离大于阈值，认为是两组

        if lines_are_far(lines[i], lines[0], tolerance_rho) and not cross_in_pic(lines[0, :, :2], lines[i, :, :2], mask_shape):
            groups[1].extend(lines[i][np.newaxis, :])
        else:
            groups[0].extend(lines[i][np.newaxis, :])
    if groups[1] == []:
        groups.pop();
    # print(groups)
    # 调整格式
    for i in range(len(groups)):
        groups[i] = np.array(groups[i])
    return groups



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
    if line1.ndim == 1:
        rho1, theta1 = line1
        rho2, theta2 = line2
    if line1.ndim == 2:
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

def get_lines_group(lines, mask_shape, tolerance_rho, visual):
    # print('get_lines_group')
    # 在每个角度组中，再按距离分组
    # 分组的等级是平等的
    theta_groups = get_lines_group_by_theta(lines, 15)
    # print('len(theta_groups)',len(theta_groups))
    # print('get out get_lines_group_by_theta')
    all_groups = []
    for theta_group in theta_groups:
        # print('going to rho')
        rho_group = get_lines_group_by_rho(theta_group, mask_shape, tolerance_rho)
        all_groups.extend(rho_group)
    all_groups = remove_none_in_list(all_groups)
    if visual:
        mask = np.zeros(mask_shape).astype('uint8')
        for i in range(len(all_groups)):
            draw_lines(all_groups[i][:, :, :2], mask, 50 + 40 * i, 2, True, 'draw_line_group')
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
    # cv2.imwrite(os.path.join(cfg.data_root, 'show', 'lab_image.png'), lab_image)
    lab_mask = cv2.adaptiveThreshold(lab_image[..., 1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 2)
    lab_mask = remove_small_area(lab_mask, 1000, False, '')
    # lav_mask_iamge= put_mask_on_img(lab_mask, lab_image, False,'lab_mask_on_image')
    # cv2.imwrite(os.path.join(cfg.data_root, 'show', 'lav_mask_iamge.png'), lav_mask_iamge)
    if color_name == 'hsv':
        color_name_image = convert_image(image, color_name)
        mask = cv2.adaptiveThreshold(color_name_image[...,1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 201, 2)
        lab_mask = get_avaliable_part(mask, lab_mask, False)
        # lav_mask_iamge = put_mask_on_img(lab_mask, color_name_image, False, 'hsv_mask_on_image')
        # cv2.imwrite(os.path.join(cfg.data_root, 'show', 'hsv_mask_iamge.png'), lav_mask_iamge)
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


def hsv_get_whole_kit(img_hsv, visual):
    kit_lower = np.array([0, 0, 0])
    kit_upper = np.array([180, 255, 46])
    mask_kit = cv2.inRange(img_hsv, kit_lower, kit_upper)
    mask_kit = largest_cc(mask_kit, True)
    if visual:
        cv2.imshow('mask_kit',mask_kit)
    return mask_kit


def get_obj_mask(image, only_upper_surface, red_space_range_dict):
    # 生成用来去噪的mask
    if only_upper_surface:
        mask_hsv = color_space_get_all_obj(image, 9, 'hsv', red_space_range_dict, False, 'hsv')
        mask_lab = color_space_get_all_obj(image, 5, 'lab', red_space_range_dict, False, 'lab')
        mask_hsv = dilate(mask_hsv, 5, 3)
        mask_lab = dilate(mask_lab, 5, 3)
    else:
        mask_hsv = adap_get_mask_in_color_space(image, 'hsv', False)
        mask_lab = adap_get_mask_in_color_space(image, 'lab', False)
        mask_hsv = dilate(mask_hsv, 3, 2)
        mask_lab = dilate(mask_lab, 3, 1)
    return mask_hsv, mask_lab

# def get_one_color_obj_mask(image, color, color_space_name):
#     #得到指定颜色的mask

def get_convinced_mask(mask_hsv,mask_lab):
    # mask_hsv = remove_surrounding_white(mask_hsv, False)
    # mask_lab = remove_surrounding_white(mask_lab, False)
    all_mask = get_union(mask_hsv, mask_lab, False, 'two_mask')
    all_mask = remove_big_area(all_mask, 30000, False, 'after remove big area')
    all_mask = remove_small_area(all_mask, 3000, False, '')
    all_mask = get_half_centroid_mask(all_mask, True, 80)
    return all_mask.astype('uint8')


def get_all_edge(image, restrict_mask, visual):
    hsv_edge = get_edge_canny(image, 'hsv', 2, 25, 45, False)
    lab_edge = get_edge_canny(image, 'lab', 0, 25, 45, False)
    all_edge = get_union(hsv_edge, lab_edge, False, 'two_edge')

    all_edge = get_intersection(all_edge, restrict_mask, False, 'all')
    # all_edge = remove_scattered_pix(all_edge, 5, False)
    all_edge = get_half_mask(all_edge,True, 70)
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

def get_point2lines_distance(points, lines):
    distance_all = []
    for line in lines:
        distance = get_point2line_distance(points, line)
        distance_all.append(distance)
    distance_all = np.array(distance_all)
    assert distance_all.ndim == 1
    return distance_all


def get_uppersurface_edge_lines(lines, lens, obj_mask, center, visual):

    def count_diff_pix(line, group_idx, groups, all_min_index):
        #DONE:保留同一组的各种diff中更小的
        all_edge = [line]
        for other_group_idx in range(len(groups)):
            if other_group_idx == group_idx:
                continue
            all_edge.append(groups[other_group_idx][all_min_index[other_group_idx]])
        # 把直线萎缩成线段
        all_edge = np.array(all_edge)
        adjacent_lines = get_adjacent_lines(all_edge[:,:,:2], obj_mask)
        vertex_coords, edge_mask = get_proper_segment_lines(adjacent_lines, obj_mask.shape, 255, -1, False)
        diff_coord = np.column_stack(np.where(np.logical_and(obj_mask == 0, edge_mask == 255)))
        return len(diff_coord)

    lines = np.concatenate((lines, lens[:,np.newaxis, np.newaxis]), axis = 2)
    groups = get_lines_group(lines, obj_mask.shape, 30, False)  # 距离大于30的平行直线会被认为是两组
    #注意得到的group.shape为(N,1,3)
    all_min_index = []
    all_group_distance = []
    for i in range(len(groups)):
        # 去除侧边缘竖线（垂直于桌面）的干扰，原理：如果一组只有一条线且当时识别他的阈值小于某值，认为是侧竖边
        if len(groups[i]) == 1 and groups[i][0,0,2] < 25:
            groups[i] = None
            continue
        group_distance = get_point2lines_distance(center, groups[i])
        all_group_distance.append(group_distance)
        min_index = np.argmin(group_distance)
        all_min_index.append(min_index)

    groups = remove_none_in_list(groups)

    assert len(all_min_index) == len(groups)
    assert len(all_min_index) == len(all_group_distance)
    # 去除外部边缘线：当一个组有两条或以上的线，对其中每条线进行判断，如果满足以下条件，则认为是底面边缘，不保留
    # 判断条件： 本线与其他组各一条直线围起来的图像中，如果有一部分在obj_mask对应位置而黑色超过一定面积，则这条线是轮廓 升级：黑色变化大于阈值，是与内部的线比较得出
    all_uppersurface_line = []
    for group_idx in range(len(groups)):
        group = groups[group_idx]
        if len(group) > 1:
            group_diff_pix = []
            for line_idx in range(len(group)):
                line = group[line_idx]
                line_diff_pix = count_diff_pix(line, group_idx, groups, all_min_index)
                group_diff_pix.append(line_diff_pix)
            # 根据group_diff_pix保留线段
            group_diff_pix = np.array(group_diff_pix)
            sorted_idx = np.argsort(group_diff_pix)
            group_diff_pix = group_diff_pix[sorted_idx]
            group = group[sorted_idx]
            # 如果开头几个是相同的diff_pix值，则分两种情况：
            #   当两者之间的距离绝对值大于10时，取其中那个距离中心更近的线，其余的忽略
            #   否则，取其中更远的线
            # 如果开头两个是不相同的值，只取diff_pix小的那条线
            if group_diff_pix[0] == group_diff_pix[1]:
                group_distance = all_group_distance[group_idx]
                group_distance = group_distance[sorted_idx]
                if group_distance[0] > group_distance[1]:
                    max_dist_line = group[0]
                    min_dist_line = group[1]
                else:
                    max_dist_line = group[1]
                    min_dist_line = group[0]
                if abs(group_distance[0] - group_distance[1]) < 7:
                    # print('have same diff_pix and they are close')
                    # print('group_distance[0] - group_distance[1]',group_distance[0] - group_distance[1])
                    # print(group_distance[0], group_distance[1])
                    all_uppersurface_line.append(max_dist_line)
                else:
                    all_uppersurface_line.append(min_dist_line)
                    # print('have same diff_pix and they are far')
                    # print('group_distance[0] - group_distance[1]', group_distance[0] - group_distance[1])
                    # print(group_distance[0], group_distance[1])

            else:
                all_uppersurface_line.append(group[0])
        else:
            all_uppersurface_line.append(group[0])
    mask = np.zeros(obj_mask.shape)
    all_uppersurface_line = np.array(all_uppersurface_line)
    draw_lines(all_uppersurface_line[:,:,:2], mask, 255, 1, False, 'after choose all uppersurface line')
    if visual:
        cv2.imshow('after choose all uppersurface ', mask)
    return all_uppersurface_line[:,:,:2], mask



# def remove_short_or_twisty(mask, line_th):
#     # 原理：去掉霍夫直线检测不出来的部分
#     _, existline_mask = detect_line_hough(mask, line_th, False)
#     mask = np.where(np.logical_and(mask == 255, existline_mask == 0), 0, mask)
#     return mask
#
# def get_all_contour_points(contours):
#     all_cnt = contours[0]
#     for i in range(1, len(contours)):
#         all_cnt = np.concatenate((all_cnt, contours[i]))
#     return all_cnt

# def get_hull_include_any(mask, visual):
#     #返回一个ndarray和其mask，代表了圈出画面内所有内容的凸包
#     contours = get_exter_contours(mask, 'none')
#     all_cnt = get_all_contour_points(contours)
#     hull = cv2.convexHull(all_cnt)
#     hull_mask = cv2.drawContours(np.zeros(mask.shape), [hull], -1, 255, 1).astype('uint8')
#     if visual:
#         cv2.imshow('get_hull_include_any', hull_mask)
#     return hull, hull_mask

# def get_approxPolyDP_inclue_any(mask, visual):
#     contours = get_exter_contours(mask, 'none')
#     all_cnt = get_all_contour_points(contours)
#     peri = cv2.arcLength(all_cnt, False)
#     approx_points = cv2.approxPolyDP(all_cnt, 0.0375 * peri, True)
#     if visual:
#         approx_mask = cv2.drawContours(np.zeros(mask.shape), [approx_points] , -1, 255, 1).astype('uint8')
#         cv2.imshow('approxPolyDP_inclue_any', approx_mask)
#
#
# def get_theta(point1,point2):
#     return np.arctan2(point2[0,1] - point1[0,1] , point2[0,0] - point1[0,0])


# def find_hough_end_point(line, mask, visual):
#     #找到一个霍夫变换的直线的端点
#     line_mask = draw_a_line(line, np.zeros(mask.shape), 255, 2)
#     # cv2.imshow('max_line_mask', max_line_mask)
#     xand_coords = np.where(np.logical_and(line_mask == mask, line_mask != 0))#找到的是相等点的所有点坐标
#     # print('xand',xand_coords)
#     if xand_coords[0] != [] and xand_coords[1] != []:#均非空
#         min_row = np.argmin(xand_coords[0])
#         max_row = np.argmax(xand_coords[0])
#         one_endpoint = (xand_coords[1][min_row],xand_coords[0][min_row])
#         another_endpoint = (xand_coords[1][max_row],xand_coords[0][max_row])
#     hough_segment_mask = cv2.line(np.zeros(mask.shape), one_endpoint, another_endpoint, 255, 1)
#     if visual:
#         cv2.imshow('find_hough_end_point',hough_segment_mask)
#     return one_endpoint, another_endpoint, hough_segment_mask


def confirm_circle(mask_list):
    circle_list = []
    for mask in mask_list:
        circles = get_convinced_circle(mask)
        if circles is None:
            continue
        elif len(circles) == 1:
            circle_list.append(circles[0])
        else:
            # 取圆心靠左的圆柱
            left_index = np.argmin(circles[:][0])
            circle_list.append(circles[left_index])
    circle_list = np.array(circle_list)

    return circle_list[np.argmin(circle_list[:, 0])]


def get_info_for_arm(image, visual, red_space_range_dict):
    # 用于输出给机械臂的信息
    # 如果visual = True，在本函数之后加上cv2.waitKey(),即可看到框出各物体上表面的绿色边缘以及物体类别
    mask_hsv, mask_lab = get_obj_mask(image, True, red_space_range_dict)
    all_mask = get_convinced_mask(mask_hsv, mask_lab)
    # cv2.imshow('all_mask', all_mask)
    # print('done union')
    # cv2.waitKey()
    each_mask_list = get_each_mask(all_mask, False)
    all_edge = get_all_edge(image, all_mask, False)
    obj_list = []  # 给机械臂的信息
    obj_dict = {}  # 用于显示图像的信息
    for each_mask in each_mask_list:
        contours = get_exter_contours(each_mask, 'simple')
        cnt = contours[0]
        mask_center = get_centroid(cnt)
        # 边缘处理
        each_edge = get_intersection(all_edge, each_mask, False, 'one obj edge')
        lab_image = (convert_image(image, 'lab')[..., 0]).astype('uint8')
        lab_image_part = get_intersection(lab_image, each_mask, False, 'lab_circle')
        circles = get_convinced_circle(lab_image_part)  # 返回的是(N,3)数组
        if circles is not None:
            # 能检测到圆形， 说明起码是有圆
            # 加上对侧边的删除
            # 圆形需要连接
            # print('yuan')
            hsv_image = (convert_image(image, 'hsv')[..., 2]).astype('uint8')
            the_circle = confirm_circle([hsv_image, lab_image])
            obj_list.append(['circle', mask_center, 0])
            obj_dict['circle'] = [(the_circle[0], the_circle[1]), 0, the_circle[2]]
        else:
            # 是正多边形
            # detect_line_hough(each_edge, 20, True)
            all_lines, all_len = adap_detect_all_line_hough(each_edge, 20, False)
            all_lines, uppersurface_mask = get_uppersurface_edge_lines(all_lines, all_len, each_mask, mask_center, False)
            # cv2.waitKey()
            # 把直线萎缩成线段
            all_lines = get_adjacent_lines(all_lines, each_mask)
            vertex_coords, edge_segment_mask = get_proper_segment_lines(all_lines, each_edge.shape, 255, 1, False)
            put_mask_on_img(edge_segment_mask, image, False, 'upper surface edge segment')
            # 从线的角度得到要旋转的角度
            geometry_shape = detect_shape_by_line_count(all_lines)
            rot_theta = get_rot_theta(edge_segment_mask, vertex_coords, False)
            x, y = get_centroid(edge_segment_mask)
            obj_list.append([geometry_shape, mask_center, rot_theta * 180 / np.pi])
            obj_dict[geometry_shape] = [(x, y), rot_theta, vertex_coords]


    if visual:
        image = draw_label_on_image(image, obj_dict, visual)
    return obj_list


def get_kit_edge(image, visual):
    kit_edge = get_edge_canny(image, 'hsv', 2, 25, 45, False)
    kit_edge = get_half_mask(kit_edge, False, 30).astype('uint8')
    hsv_image = convert_image(image, 'hsv')
    kit_mask = hsv_get_whole_kit(hsv_image, False)
    kit_mask = dilate(kit_mask, 3, 1)
    kit_edge = get_intersection(kit_edge, kit_mask, False, 'of kit')
    all_kit_lines, all_kit_linelen = adap_detect_all_line_hough(kit_edge, 100, visual)
    return all_kit_lines, all_kit_linelen

def get_the_kit_loc_line(kit_lines):
    #分别取出最左边的垂直竖线(侧面露出来的底线)和最下方的水平横线 用来定位盒子
    vertical_line = kit_lines[np.where(np.logical_or(np.pi - kit_lines[..., 1] < 10 * np.pi / 180, kit_lines[..., 1] < 10 * np.pi / 180))]
    vertical_line = vertical_line[np.argmin(np.abs(vertical_line[..., 0]))]
    hori_line = kit_lines[np.where(np.abs(np.pi /2 - kit_lines[..., 1]) < 10 * np.pi / 180)]
    hori_line = hori_line[np.argmax(np.abs(hori_line[..., 0]))]
    return vertical_line, hori_line

def get_kit_offset(now_kit, ref_kit):
    now_kit_lines, now_kit_linelen = get_kit_edge(now_kit, False)

    now_verti_line, now_hori_line = get_the_kit_loc_line(now_kit_lines)
    now_left_bottom = get_cross_point(now_verti_line, now_hori_line)
    ref_kit_lines, ref_kit_linelen = get_kit_edge(ref_kit, False)
    ref_verti_line, ref_hori_line = get_the_kit_loc_line(ref_kit_lines)
    ref_left_bottom = get_cross_point(ref_verti_line, ref_hori_line)
    offset_x = now_left_bottom[0] - ref_left_bottom[0]
    offset_y = now_left_bottom[1] - ref_left_bottom[1]
    # 分别返回现在相对ref_kit中盒子的位移,向右和向下为正
    print(offset_x, offset_y)
    return offset_x, offset_y


def adap_mask_one_channel_tool(image, need_save, i, visual):
    # hsv的1通道经过adaptive之后可以很好的分割出整个物体，不要inv，但是还需要各种后处理
    # hsv的2通道在canny下可以得到不错的边缘，但是自适应分割时完全看不出

    # lab的2通道在adaptive分割之后可以分割出整个物体，要inv，保留左半边并去除小区域即可
    # lab 0 通道可用于canny得到两个表面的边缘
    def get_edge(onechannel_image, th1, th2):
        onechannel_image = cv2.GaussianBlur(onechannel_image, ksize=(3,3), sigmaX=0, dst=None, sigmaY=None, borderType=None)
        onechannel_image = cv2.medianBlur(onechannel_image, 3)
        color_name_edges = cv2.Canny(onechannel_image, th1, th2)
        return color_name_edges

    # #rgb三个通道分别处理,以及变成灰度
    # rgb0_mask = cv2.adaptiveThreshold(image[..., 0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 201, 2)
    # rgb1_mask = cv2.adaptiveThreshold(image[..., 1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 201, 2)
    # rgb2_mask = cv2.adaptiveThreshold(image[..., 2], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 201, 2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 201, 2)

    #HSV三个
    hsv_image = convert_image(image, 'hsv')
    # hsv0_mask = cv2.adaptiveThreshold(hsv_image[..., 0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 2)
    # hsv1_mask = cv2.adaptiveThreshold(hsv_image[..., 1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 2)
    # hsv2_mask = cv2.adaptiveThreshold(hsv_image[..., 2], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 201, 2)

    #Lab三个
    lab_image = convert_image(image, 'lab')
    # lab0_mask = cv2.adaptiveThreshold(lab_image[..., 0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 201, 2)
    # lab1_mask = cv2.adaptiveThreshold(lab_image[..., 1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 2)
    # lab2_mask = cv2.adaptiveThreshold(lab_image[..., 2], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 201, 2)

    # 对边缘的检测
    rgb0_edge = get_edge(image[..., 0], 5, 10)
    rgb1_edge = get_edge(image[..., 1], 5, 10)
    rgb2_edge = get_edge(image[..., 2], 5, 10)
    gray_edge = get_edge(gray, 5, 10)

    hsv0_edge = get_edge(hsv_image[..., 0], 5, 10)
    hsv1_edge = get_edge(hsv_image[..., 1], 5, 10)
    hsv2_edge = get_edge(hsv_image[..., 2], 5, 10)

    lab0_edge = get_edge(lab_image[..., 0], 5, 10)
    lab1_edge = get_edge(lab_image[..., 1], 5, 10)
    lab2_edge = get_edge(lab_image[..., 2], 5, 10)

    if need_save:
        # cv2.imwrite(os.path.join(img_dir, 'color{}_rgb0_mask.png'.format(i)), rgb0_mask)
        # cv2.imwrite(os.path.join(img_dir, 'color{}_rgb1_mask.png'.format(i)), rgb1_mask)
        # cv2.imwrite(os.path.join(img_dir, 'color{}_rgb2_mask.png'.format(i)), rgb2_mask)
        # cv2.imwrite(os.path.join(img_dir, 'color{}_rgb3_mask.png'.format(i)), gray_mask)
        #
        # cv2.imwrite(os.path.join(img_dir, 'color{}_hsv0_mask.png'.format(i)), hsv0_mask)
        # cv2.imwrite(os.path.join(img_dir, 'color{}_hsv1_mask.png'.format(i)), hsv1_mask)
        # cv2.imwrite(os.path.join(img_dir, 'color{}_hsv2_mask.png'.format(i)), hsv2_mask)
        #
        # cv2.imwrite(os.path.join(img_dir, 'color{}_lab0_mask.png'.format(i)), lab0_mask)
        # cv2.imwrite(os.path.join(img_dir, 'color{}_lab1_mask.png'.format(i)), lab1_mask)
        # cv2.imwrite(os.path.join(img_dir, 'color{}_lab2_mask.png'.format(i)), lab2_mask)


        cv2.imwrite(os.path.join(img_dir, 'color{}_rgb0_edge.png'.format(i)), 255- rgb0_edge)
        cv2.imwrite(os.path.join(img_dir, 'color{}_rgb1_edge.png'.format(i)), 255- rgb1_edge)
        cv2.imwrite(os.path.join(img_dir, 'color{}_rgb2_edge.png'.format(i)), 255- rgb2_edge)
        cv2.imwrite(os.path.join(img_dir, 'color{}_rgb3_edge.png'.format(i)), 255- gray_edge)

        cv2.imwrite(os.path.join(img_dir, 'color{}_hsv0_edge.png'.format(i)), 255- hsv0_edge)
        cv2.imwrite(os.path.join(img_dir, 'color{}_hsv1_edge.png'.format(i)), 255- hsv1_edge)
        cv2.imwrite(os.path.join(img_dir, 'color{}_hsv2_edge.png'.format(i)), 255- hsv2_edge)

        cv2.imwrite(os.path.join(img_dir, 'color{}_lab0_edge.png'.format(i)), 255- lab0_edge)
        cv2.imwrite(os.path.join(img_dir, 'color{}_lab1_edge.png'.format(i)), 255- lab1_edge)
        cv2.imwrite(os.path.join(img_dir, 'color{}_lab2_edge.png'.format(i)), 255- lab2_edge)




    # if visual:
    #     cv2.imshow('rgb0_mask', rgb0_mask)
    #     cv2.imshow('rgb1_mask', rgb1_mask)
    #     cv2.imshow('rgb2_mask', rgb2_mask)
    #     cv2.imshow('gray_mask', gray_mask)
    #
    #     cv2.imshow('hsv0_mask', hsv0_mask)
    #     cv2.imshow('hsv1_mask', hsv1_mask)
    #     cv2.imshow('hsv2_mask', hsv2_mask)
    #
    #     cv2.imshow('lab0_mask', lab0_mask)
    #     cv2.imshow('lab1_mask', lab1_mask)
    #     cv2.imshow('lab2_mask', lab2_mask)
    #
    #     cv2.waitKey()

def adap_mask_by_saturability(image, visual):
    #实验发现hsv1最能有效地分割出物体，hsv通道对应颜色的饱和度
    hsv_image = convert_image(image, 'hsv')
    hsv1_mask = cv2.adaptiveThreshold(hsv_image[..., 1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 2)
    # 去噪,会损失顶点上的尖端
    hsv1_mask = remove_surrounding_white(hsv1_mask, False) #去掉左右与边缘连接的白色部分，这要求物体一定不能接触画面边缘，不然会被当做噪声去掉
    hsv1_mask = erode(hsv1_mask, 3, 2)
    hsv1_mask = remove_big_area(hsv1_mask, 10000, False, 'hsv1')
    hsv1_mask = get_half_mask(hsv1_mask, True, 100)
    hsv1_mask = remove_small_area(hsv1_mask, 2000, False, 'hsv1')
    hsv1_mask = dilate(hsv1_mask, 3, 2)
    if visual:
        cv2.imshow( "hsv1_mask after remove noise", hsv1_mask)
    return hsv1_mask

def get_color_acc2_coord(image, coord):
    # 获得xy坐标选中的部分的图形颜色（假设每个mask中只有一种颜色）
    one_pixel_color_bgr = image[coord[1], coord[0]]
    one_pixel_color_bgr =one_pixel_color_bgr[np.newaxis, np.newaxis,:]
    one_pixel_color_hsv = cv2.cvtColor(one_pixel_color_bgr, cv2.COLOR_BGR2HSV)
    print('当前mask的像素点，其bgr颜色为 {}， hsv颜色为{}'.format(one_pixel_color_bgr, one_pixel_color_hsv))


if __name__ == "__main__":
    #全蓝色的路径
    # top_dir = os.path.join(cfg.data_root, cfg.data_type)
    # img_dir = os.path.join(cfg.data_root,'img')
    #多色路径
    img_dir = os.path.join('various_color_data')
    # points_dit = os.path.join(cfg.data_root,'points')
    rgb_image_num = len(os.listdir(img_dir))#//2
    pixel_range = []
    dict = {}
    need_update= False
    select_param = False
    # color_space_range_dict = {}
    blue_space_range_dict = {'obj_hsv_lower': np.array([ 1, 94, 77]), 'obj_hsv_upper': np.array([4, 255, 158]),                     #'obj_hsv_lower': np.array([101, 93, 90]), 'obj_hsv_upper': np.array([106, 255, 142])
                              'obj_ycrcb_lower': np.array([40, 87, 137]), 'obj_ycrcb_upper': np.array([103, 115, 164]),
                              'obj_xyz_lower': np.array([35, 43, 93]), 'obj_xyz_upper': np.array([104, 113, 150]),
                              'obj_hls_lower': np.array([101, 46, 65]), 'obj_hls_upper': np.array([107, 116, 255]),
                              'obj_lab_lower': np.array([53, 121, 94]), 'obj_lab_upper': np.array([113, 132, 108]),
                              'obj_luv_lower': np.array([52, 81, 93]), 'obj_luv_upper': np.array([113, 88, 111])
                              }
    red_space_range_dict = {'obj_hsv_lower': np.array([ 0, 77, 51]), 'obj_hsv_upper': np.array([ 14, 253, 183]),
                            'obj_lab_lower': np.array([ 29, 137, 129]), 'obj_lab_upper': np.array([174, 179, 167])
                            }
    black_space_range_dict = {'obj_hsv_lower': np.array([ 0,  0, 12]), 'obj_hsv_upper': np.array([ 80, 149,  41])}

    # for i in range(1, 18, 2):
    #     dict['{}'.format(i)] = 0

    color_space_name = 'lab'
    for i in range(5,rgb_image_num):#range(7,rgb_image_num,8)
        # if not os.path.exists(os.path.join(img_dir, 'color{}.png'.format(i))):
        #     continue
        # else:
        if os.path.exists(os.path.join(img_dir,'color{}.png'.format(i))):
            image = cv2.imread(os.path.join(img_dir, 'color{}.png'.format(i)))
            # cv2.imshow('original_image',image)
            print('=============={}==================='.format(i))
            # adap_mask_one_channel_tool(image, True, i, False)



            obj_list = get_info_for_arm(image, True)
            # print(obj_list)
            # ref_kit = cv2.imread(os.path.join(cfg.data_root,'img','ref_kit.jpg'))
            # get_kit_offset(image, ref_kit)
            cv2.waitKey()

            # w, h, c = image.shape
            # half_w = h//2
            #
            # #hsv的1通道经过adaptive之后可以很好的分割出整个物体，不要inv，但是还需要各种后处理
            # #hsv的2通道在canny下可以得到不错的边缘，但是自适应分割时完全看不出
            #
            # #lab的2通道在adaptive分割之后可以分割出整个物体，要inv，保留左半边并去除小区域即可
            # #lab 0 通道可用于canny得到两个表面的边缘
            # mask_hsv1 = adap_mask_by_saturability(image, False)
            # each_mask_list = get_each_mask(mask_hsv1, False)
            # for each_mask in each_mask_list:
            #     # cv2.imshow('eache_mask', each_mask)
            #     contours = get_exter_contours(each_mask, 'simple')
            #     cnt = contours[0]
            #     mask_center = get_centroid(cnt)
            #     assert mask_center is not None
            #     get_color_acc2_coord(image,mask_center)
            #     # cv2.waitKey()

        #     mask_hsv,mask_lab = get_obj_mask(image, False)
        #
        #     all_mask = get_convinced_mask(mask_hsv,mask_lab)
        #     cv2.imshow('all_mask', all_mask)
        #
        #
        #     each_mask_list = get_each_mask(all_mask, False)
        #     all_edge = get_all_edge(image, all_mask, True)
        #     cv2.waitKey()
        #     show_point_img = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2BGR)
        #     obj_dict = {}
        #     for each_mask in each_mask_list:
        #         contours = get_exter_contours(each_mask, 'simple')
        #         cnt = contours[0]
        #         mask_center = get_centroid(cnt)
        #         # 边缘处理
        #         each_edge = get_intersection(all_edge, each_mask, False, 'one obj edge')
        #         lab_image = (convert_image(image, 'lab')[...,0]).astype('uint8')
        #         lab_image_part = get_intersection(lab_image, each_mask, False, 'lab_circle')
        #         circles = get_convinced_circle(lab_image_part)  # 返回的是(N,3)数组
        #         if circles is not None:
        #             # 能检测到圆形， 说明起码是有圆
        #             # 加上对侧边的删除
        #             # 圆形需要连接
        #             hsv_image = (convert_image(image, 'hsv')[..., 2]).astype('uint8')
        #             the_circle = confirm_circle([hsv_image, lab_image])
        #             obj_dict['circle'] = [(the_circle[0], the_circle[1]), 0, the_circle[2]]
        #
        #             point_mask = np.zeros(each_mask.shape)
        #
        #             # imgs = cv2.cvtColor(each_mask, cv2.COLOR_GRAY2BGR)
        #             cv2.circle(show_point_img, mask_center, 10, (0,0,255), -1)
        #
        #             # cv2.imwrite(os.path.join(cfg.data_root, 'show', 'circle_edge.png'), each_edge)
        #             # cv2.imwrite(os.path.join(cfg.data_root, 'show', '{}_edge_segment.png').format(geometry_shape),
        #             #             edge_segment_mask)
        #             # cv2.imwrite(os.path.join(cfg.data_root, 'show', 'circle_loc.png'), imgs)
        #             # cv2.waitKey()
        #         else:
        #             # 是正多边形
        #             print('无圆')
        #             # detect_line_hough(each_edge, 20, True)
        #
        #             all_lines, all_len = adap_detect_all_line_hough(each_edge, 20, False)
        #             all_lines, uppersurface_mask = get_uppersurface_edge_lines(all_lines, all_len, each_mask, mask_center, False)
        #             # 把直线萎缩成线段
        #             all_lines = get_adjacent_lines(all_lines, each_mask)
        #             vertex_coords, edge_segment_mask = get_proper_segment_lines(all_lines, each_edge.shape, 255, 1, False)
        #             put_mask_on_img(edge_segment_mask, image, False, 'upper surface edge segment')
        #             # 从线的角度得到要旋转的角度
        #             geometry_shape = detect_shape_by_line_count(all_lines)
        #             # rot_theta = get_rot_theta_new(vertex_coords, center, (w, h), edge_segment_mask, True)
        #             rot_theta = get_rot_theta(edge_segment_mask, vertex_coords, False)
        #             x,y = get_centroid(edge_segment_mask)
        #             obj_dict[geometry_shape] = [(x, y), rot_theta, vertex_coords]
        #
        #             # cv2.imwrite(os.path.join(cfg.data_root, 'show', '{}_edge.png').format(geometry_shape), each_edge)
        #             # cv2.imwrite(os.path.join(cfg.data_root, 'show', '{}_edge_segment.png').format(geometry_shape), edge_segment_mask)
        #
        #             # point_mask = np.zeros(each_mask.shape)
        #             # cv2.circle(point_mask, mask_center, 10, 255, 2)
        #             # mask = put_mask_on_img(point_mask, each_mask, False, '')
        #             # cv2.imwrite(os.path.join(cfg.data_root, 'show', '{}_loc.png').format(geometry_shape), mask)
        #
        #
        #             # cv2.circle(show_point_img, mask_center, 10, (0, 0, 255), -1)
        #     # cv2.imwrite(os.path.join(cfg.data_root, 'show', 'all_loc.png'), show_point_img)
        #
        #             # cv2.waitKey()
        #     if obj_dict:
        #         image = draw_label_on_image(image,obj_dict, True)
        #
        #
        #     # 保存检测图片和信息字典
        #     # np.save(os.path.join(cfg.data_root, 'label', 'objs_{}.npy'.format(i)), obj_dict)
        #     # cv2.imwrite(os.path.join(cfg.data_root, 'label', 'objs_{}.png'.format(i)), image)
        #
        #
        #     cv2.waitKey()
        #     #
        #     if select_param:
        #         compare_corners(image, dict, i, color_space_name, color_space_range_dict)
        #
            if need_update:
                pixel_range = update_mask(image,pixel_range,color_space_name)
        if need_update:
            print('======new_range====',pixel_range)
            # color_space_range_dict['obj_{}_lower'.format(color_space_name)] = pixel_range[0]
            # color_space_range_dict['obj_{}_upper'.format(color_space_name)] = pixel_range[1]
            # print(color_space_range_dict)


        # print(dict)






