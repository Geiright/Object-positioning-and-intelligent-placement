import cv2
from skimage.measure import label
import numpy as np
from form2fit.code.utils.misc import largest_cc,mask2bbox

def remove_small_area(mask,area_th):
    print(type(mask),mask.shape)
    contours, hierarch = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < area_th:
            cv2.drawContours(mask, [contours[i]], 0, 0, -1)
    return mask

def remove_slim(mask,ratio = 5):
    '''
    去掉细长的非0部分
    :param mask:
    :param ratio: 越细长，连通域的面积/周长就越小
    :return:
    '''
    contours, hierarch = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(len(contours))
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        perimeter = cv2.arcLength(contours[i], True)
        print(area)
        print(perimeter)
        if area < perimeter *ratio:
            cv2.drawContours(mask, [contours[i]], 0, 0, -1)
    return mask

def remove_surrounding_white(mask,visual):
    '''
    在mask中，去掉贴着图像边缘的白色部分（通常是背景）
    :param mask:
    :param visual: 可视化
    :return: mask：处理后的mask
    '''
    h,w = mask.shape
    labels = label(mask)
    if visual:
        cv2.imshow('labels going to remove_surrounding_white',(labels*40).astype('uint8'))
    num = np.max(labels)
    if visual:
        print('num in remove_surrounding_white',num)
    if num > 1:#如果只有一个连通域，不需要处理
        for i in range(num):
            domain = np.where(labels==i+1,1,0)
            if visual:
                cv2.imshow('domain in remove_surrounding_white',(domain.astype('uint8'))*255)
            rmin,rmax,cmin,cmax = mask2bbox(domain)
            if rmin ==0 or rmax == h-1 or cmin == 0 or cmax == w-1:
                labels = np.where(labels == i+1 , 0 , labels)
        mask = np.where(labels !=0,mask,0)
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
    if visual:
        print('num in remove_inner_black', num)
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
        cv2.imshow('mask in remove_inner_white', mask)
    return mask



def adap_get_desk(dimg,cimg, ref_type,visual):
    '''
    以深度图或者灰度图为参考，自适应去掉背景，保留桌面及上面的物体
    :param dimg: 深度图
    :param cimg: 灰度图
    :param ref_type: 为0时，以深度图为参考，为1时，以灰度图为参考
    :param visual:为真时，处理过程和结果可视化
    :return: dimg,cimg：去掉背景之后的深度图dimg和灰度图cimg
    '''
    # dimg_i = cv2.imread(os.path.join(dirname, 'init_depth_height.png'), cv2.IMREAD_GRAYSCALE)
    # dimg_f = cv2.imread(os.path.join(dirname, 'final_depth_height.png'), cv2.IMREAD_GRAYSCALE)
    # cimg_i = cv2.imread(os.path.join(dirname, 'init_color_height.png'), cv2.IMREAD_GRAYSCALE)
    # cimg_f = cv2.imread(os.path.join(dirname, 'final_color_height.png'), cv2.IMREAD_GRAYSCALE)

    if ref_type == 0:
        thre_otsu,seg_img  = cv2.threshold(dimg ,0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if visual:
            cv2.imshow('seg_img', seg_img)
        seg_img = largest_cc(seg_img, True)
        seg_img_nosr = remove_surrounding_white(seg_img, visual)
        seg_img_noin = remove_inner_black(seg_img_nosr,visual)
        if visual:
            cv2.imshow('desk_mask', seg_img)
            cv2.imshow('seg_img_nosr', seg_img_nosr)
            cv2.imshow('seg_img_noin', seg_img_noin)
    else:
        seg_img = cv2.adaptiveThreshold(cimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 801, 2)
        if visual:
            cv2.imshow('seg_img', seg_img)
        seg_img = largest_cc(seg_img,True)
        seg_img_nosr = remove_surrounding_white(seg_img,visual)
        seg_img_noin = remove_inner_black(seg_img_nosr,visual)
        if visual:
            cv2.imshow('desk_mask',seg_img)
            cv2.imshow('seg_img_nosr',seg_img_nosr)
            cv2.imshow('seg_img_noin', seg_img_noin)
    dimg = np.where(seg_img_noin,dimg,0)
    cimg = np.where(seg_img_noin,cimg,0)
    if visual:
        cv2.imshow('dimg_i',dimg)
        cv2.imshow('cimg_i', cimg)
    return dimg,cimg


def suction_background_substract(c_height_i,d_height_i,c_height_f,d_height_f):
    #吸取网络在训练和推断的共有步骤

    #对盒子分割
    _, img_otsu = cv2.threshold(d_height_i,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    bool_otsu = largest_cc(img_otsu)#largest_cc返回值是一个bool类型的矩阵
    mask_kit = np.zeros_like(d_height_i)
    rmin,rmax,cmin,cmax = mask2bbox(bool_otsu)
    mask_kit[rmin:rmax,cmin:cmax] = 1
    c_height_i = np.where(mask_kit, c_height_i, 0)
    d_height_i = np.where(mask_kit, d_height_i, 0)
    #对物体分割
    mask_obj = cv2.adaptiveThreshold(d_height_f, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,401, 2)
    mask_obj = 255 - mask_obj
    mask_obj = remove_small_area(mask_obj,500)

    assert mask_obj.shape == d_height_f.shape
    c_height_f = np.where(mask_obj,0,c_height_f)
    d_height_f = np.where(mask_obj,0,d_height_f)
    return c_height_i,d_height_i,c_height_f,d_height_f