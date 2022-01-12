import cv2
import numpy as np
from walle.core import RotationMatrix


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


def c2b(M, cam_points):                     # camera  to  board
    assert cam_points.shape == (1,2)
    assert M.shape == (2,3)
    R = M[:,:2]
    T = M[:,2]
    cam_points = np.float32(cam_points)
    board_points = (M @ np.hstack((cam_points, np.ones((len(cam_points), 1)))).T).T
    return board_points

def b2c(Mn, board_points):                   # board  to  camera
    assert Mn.shape == (2,3)
    board_points = np.array(board_points, dtype='float32')
    board_points = np.expand_dims(board_points, axis=0)
    # Mn = np.linalg.inv(M) # 求逆
    cam_points = (Mn @ np.hstack((board_points, np.ones((len(board_points), 1)))).T).T
    return cam_points.tolist()

def getM():                             # 物体部分的坐标
    imgpoints = [[563,615], [381,619], [214,623], [211,477], [370,469], [545,461], [230,225]]
    objpoints = [[214.4791, -46.4357], [212.3297, -105.9849], [210.5402, -159.6319], [162.6528, -159.6316], [162.6527, -108.4162], [162.6526, -50.669], [80.1705, -150.8045]]
    imgpoints = np.array(imgpoints,dtype='float32')
    objpoints = np.array(objpoints,dtype='float32')
    M, _ = cv2.estimateAffine2D(imgpoints, objpoints,True)      # 第二种方式：九点标定法
    Mn, _ = cv2.estimateAffine2D(objpoints, imgpoints,True)
    return M, Mn                                         # M是从像素坐标系转换为机械臂坐标系的矩阵，Mn反之

def getM_box():                             # 盒子部分的坐标
    imgpoints = [[893,261], [889,431], [1079,428], [1079,264]]
    objpoints = [[98.8834, 71.6999], [157.0539, 67.6597], [158.7592,132.9691], [102.5297,135.2645]]
    imgpoints = np.array(imgpoints,dtype='float32')
    objpoints = np.array(objpoints,dtype='float32')
    M, _ = cv2.estimateAffine2D(imgpoints, objpoints,True)      # 第二种方式：九点标定法
    Mn, _ = cv2.estimateAffine2D(objpoints, imgpoints,True)
    return M, Mn                                         # M是从像素坐标系转换为机械臂坐标系的矩阵，Mn反之

if __name__ == '__main__':
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

    


    M, Mn = getM()         
    pts3 = np.float32([[250,563]])
    
    angle = rotz2angle(M)
    angle = 180 * angle / 3.1415926
    print("angle = ",int(angle))
    
    #value_after_rot = (M @ np.hstack((pts3, np.ones((len(pts3), 1)))).T).T
    bpoints = c2b(M, pts3)
    print(bpoints)
    pts3 = b2c(Mn, bpoints)
    print(pts3)