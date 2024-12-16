import numpy as np
import cv2
import matplotlib.pyplot as plt
from compute_fundamental_matrix import compute_fundamental_matrix_with_RANSAC, select_corresponding_points

def compute_projection_matrices(F):
    """計算投影矩陣 P 和 P'"""
    # P = [I|0] for first camera
    P1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])
    
    # 從基本矩陣計算本質矩陣
    # 假設相機內參為單位矩陣（如果有實際的相機內參，應該使用真實值）
    E = F
    
    # 從本質矩陣分解得到 R 和 t
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])
    
    # 可能的 R 和 t
    R = U @ W @ Vt
    t = U[:, 2]
    
    # P' = [R|t]
    P2 = np.hstack((R, t.reshape(3, 1)))
    
    return P1, P2

def triangulate_points(P1, P2, points1, points2):
    """使用 DLT 進行三角測量"""
    points_3d = []
    for pt1, pt2 in zip(points1, points2):
        A = np.zeros((4, 4))
        A[0] = pt1[0] * P1[2] - P1[0]
        A[1] = pt1[1] * P1[2] - P1[1]
        A[2] = pt2[0] * P2[2] - P2[0]
        A[3] = pt2[1] * P2[2] - P2[1]
        
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]  # 齊次座標正規化
        points_3d.append(X[:3])
    
    return np.array(points_3d)

def compute_reprojection_error(points_3d, points_2d, P):
    """計算重投影誤差"""
    errors = []
    for X, x in zip(points_3d, points_2d):
        # 將 3D 點投影到 2D
        X_h = np.append(X, 1)
        x_proj = P @ X_h
        x_proj = x_proj[:2] / x_proj[2]
        
        # 計算誤差
        error = np.linalg.norm(x_proj - x)
        errors.append(error)
    
    return np.mean(errors)

def plot_3d_points(points_3d, title="3D Points"):
    """繪製 3D 點雲"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()

def main():
    # 讀取圖片
    img1 = cv2.imread('hw9/images/homography_images/image_03.jpg')
    img2 = cv2.imread('hw9/images/homography_images/image_02.jpg')
    
    # 選取對應點
    points1, points2 = select_corresponding_points(img1, img2)
    print("Selected points:")
    # type
    # <class 'numpy.ndarray'>
    print("Image 1:\n", points1)
#      [[ 748.97801063  795.17532468]
#  [ 920.39359504  797.73376623]
#  [1086.69229634  800.29220779]
#  [1242.7572314   790.05844156]
#  [ 759.21177686  958.91558442]
#  [ 935.74424439  964.03246753]
#  [1099.48450413  948.68181818]
#  [1270.90008855  953.7987013 ]]
    print("Image 2:\n", points2)
#      [[ 746.07069067  879.6038961 ]
#  [ 894.46030106  879.6038961 ]
#  [1032.61614522  884.72077922]
#  [1178.44731405  879.6038961 ]
#  [ 753.74601535  987.05844156]
#  [ 912.36939197  989.61688312]
#  [1058.2005608   981.94155844]
#  [1209.14861275  979.38311688]]
    
    # 計算基本矩陣
    F, inliers = compute_fundamental_matrix_with_RANSAC(points1, points2)
    print("Fundamental Matrix:\n", F)
#      [[ 4.17196738e-08 -1.11960870e-06  9.66049773e-04]
#  [ 1.47435149e-06  6.62930295e-07 -2.07657194e-03]
#  [-1.41971221e-03  4.72017506e-04  1.00000000e+00]]
    
    # 1. 投影重建
    P1, P2 = compute_projection_matrices(F)
    points_3d = triangulate_points(P1, P2, points1, points2)
    
    # 計算重投影誤差
    error1 = compute_reprojection_error(points_3d, points1, P1)
    error2 = compute_reprojection_error(points_3d, points2, P2)
    
    print("Projective Reconstruction:")
    print("P1:\n", P1)
    print("P2:\n", P2)
    print(f"Reprojection errors: {error1:.3f}, {error2:.3f}")
    
    # 繪製 3D 點雲
    plot_3d_points(points_3d, "Projective Reconstruction")
    
    # 2. 仿射重建
    # TODO: 實作仿射升級
    
    # 3. 度量重建
    # TODO: 實作度量升級

if __name__ == "__main__":
    main()