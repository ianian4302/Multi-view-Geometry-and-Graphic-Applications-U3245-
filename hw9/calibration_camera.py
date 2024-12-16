import glob
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 8行11列棋盘角点
CHECKERBOARD = (5, 7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 世界坐标中的3D角点，z恒为0
objpoints = []
# 像素坐标中的2D点
imgpoints = []

# 利用棋盘定义世界坐标系中的角点
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# 从文件夹中读取所有图片
images = glob.glob('hw9/images/camera_calibration_images/*.jpg')
gray = None
for i in range(len(images)):
    fname = images[i]
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 儲存灰階圖片
    cv2.imwrite('hw9/images/gray_{}.jpg'.format(i), gray)
    # 查找棋盤角點
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    print(ret)
    """
    使用cornerSubPix优化探测到的角点
    """
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # 显示角点
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        new_img = Image.fromarray(img.astype(np.uint8))
        new_img.save('chessboard_{}.png'.format(i))
        plt.imshow(img)
        plt.show()
    cv2.imshow('img', img)
    cv2.waitKey(0)

# cv2.destroyAllWindows()
# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("重投影误差:\n")
print(ret)
print("内参 : \n")
print(mtx)
print("畸变 : \n")
print(dist)
print("旋转向量 : \n")
print(rvecs)
print("平移向量 : \n")
print(tvecs)

# 重投影误差:
# 0.8164415580952856

# 内参 :
# [[1.74001048e+03 0.00000000e+00 1.19843037e+03]
#  [0.00000000e+00 1.73585082e+03 8.57254549e+02]
#  [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

# 畸变
# [[ 2.02319355e-01 -9.87416607e-01 -2.12596218e-04 -2.70232104e-04
#    1.64145247e+00]]

# 旋转向量
# (array([[-0.00322263],
#        [ 0.36248057],
#        [ 1.50552787]]), array([[ 0.37250442],
#        [-0.04303642],
#        [ 1.58986679]]), array([[-0.36273058],
#        [-0.49855888],
#        [ 1.57008014]]), array([[-0.13907297],
#        [-0.32400169],
#        [ 1.57950719]]), array([[-0.42317362],
#        [-0.30273564],
#        [ 1.60171401]]), array([[-0.01568742],
#        [-0.05727704],
#        [ 1.57562271]]))

# 平移向量
# (array([[ 2.14059296],
#        [-2.36947278],
#        [ 7.9428996 ]]), array([[ 2.14074163],
#        [-1.41728382],
#        [ 7.17939566]]), array([[ 3.6690291 ],
#        [-1.79881672],
#        [ 9.41058635]]), array([[ 3.7211247 ],
#        [-1.85146775],
#        [ 8.3413088 ]]), array([[ 3.04192939],
#        [-1.89187397],
#        [10.22361611]]), array([[ 3.04531107],
#        [-1.59460322],
#        [ 7.29729811]]))