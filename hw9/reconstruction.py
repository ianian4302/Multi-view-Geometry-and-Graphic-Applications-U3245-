import numpy as np
import cv2
import matplotlib.pyplot as plt
from compute_fundamental_matrix import compute_fundamental_matrix_with_RANSAC, select_corresponding_points

def compute_projection_matrices(F):
    """計算投影矩陣 P 和 P'"""
    # 確保F是2D數組
    F = np.array(F).reshape(3, 3)
    
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
    
    # 確保R是旋轉矩陣（行列式為1）
    if np.linalg.det(R) < 0:
        R = -R
        t = -t
    
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

def upgrade_to_affine(P1, P2, points_3d):
    """將投影重建升級為仿射重建"""
    
    def estimate_plane_at_infinity():
        """估計無窮遠平面"""
        # 構建線性方程組來求解無窮遠平面
        n_points = len(points_3d)
        A = np.zeros((n_points, 4))
        
        # 使用平行性約束
        for i in range(n_points):
            X = np.append(points_3d[i], 1)  # 齊次座標
            A[i] = X
            
        # 使用SVD求解最小二乘解
        _, _, Vt = np.linalg.svd(A)
        pi_infinity = Vt[-1]  # 最小奇異值對應的向量
        return pi_infinity
    
    def compute_affine_transformation(pi_infinity):
        """計算仿射變換矩陣"""
        # 構建仿射變換矩陣
        Ha = np.eye(4)
        Ha[3, :] = pi_infinity
        
        return Ha
    
    # 1. 估計無窮遠平面
    pi_infinity = estimate_plane_at_infinity()
    
    # 2. 計算仿射變換矩陣
    Ha = compute_affine_transformation(pi_infinity)
    
    # 3. 更新3D點
    points_3d_affine = []
    for X in points_3d:
        X_h = np.append(X, 1)
        X_a = Ha @ X_h
        X_a = X_a[:3] / X_a[3]
        points_3d_affine.append(X_a)
    
    # 4. 更新相機矩陣
    P1_affine = P1 @ np.linalg.inv(Ha)
    P2_affine = P2 @ np.linalg.inv(Ha)
    
    return np.array(points_3d_affine), P1_affine, P2_affine, Ha

def upgrade_to_metric(P1_affine, P2_affine, points_3d_affine):
    """將仿射重建升級為度量重建"""
    
    def estimate_IAC():
        """估計絕對二次曲線 (Image of Absolute Conic)"""
        # 構建約束矩陣
        n_equations = 5
        A = np.zeros((n_equations, 6))
        
        # 從相機矩陣提取仿射變換部分
        A1 = P1_affine[:, :3]
        A2 = P2_affine[:, :3]
        
        def add_orthogonality_constraint(A_mat, row_idx):
            i1 = A_mat[0]
            i2 = A_mat[1]
            i3 = A_mat[2]
            
            # i1^T * i2 = 0
            A[row_idx] = [i1[0]*i2[0], i1[0]*i2[1] + i1[1]*i2[0], 
                         i1[1]*i2[1], i1[2]*i2[0], i1[2]*i2[1], i1[2]*i2[2]]
            
            # ||i1|| = ||i2||
            A[row_idx+1] = [i1[0]**2 - i2[0]**2, 
                           2*i1[0]*i1[1] - 2*i2[0]*i2[1],
                           i1[1]**2 - i2[1]**2,
                           2*i1[0]*i1[2] - 2*i2[0]*i2[2],
                           2*i1[1]*i1[2] - 2*i2[1]*i2[2],
                           i1[2]**2 - i2[2]**2]
        
        # 添加兩個相機約束
        add_orthogonality_constraint(A1, 0)
        add_orthogonality_constraint(A2, 2)
        
        # 求解線性方程組
        _, _, Vt = np.linalg.svd(A)
        omega = Vt[-1]
        
        # 重構 3x3 對稱矩陣
        Omega = np.array([[omega[0], omega[1], omega[3]],
                         [omega[1], omega[2], omega[4]],
                         [omega[3], omega[4], omega[5]]])
        
        # 確保矩陣是正定的
        eigenvals, eigenvecs = np.linalg.eigh(Omega)
        eigenvals = np.abs(eigenvals)  # 使所有特徵值為正
        Omega = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        return Omega
    
    def compute_metric_transformation(Omega):
        """計算度量變換矩陣"""
        try:
            # 嘗試 Cholesky 分解
            L = np.linalg.cholesky(Omega)
        except np.linalg.LinAlgError:
            # 如果 Cholesky 分解失敗，使用特徵值分解
            eigenvals, eigenvecs = np.linalg.eigh(Omega)
            # 確保所有特徵值為正
            eigenvals = np.abs(eigenvals)
            # 計算等效的分解
            L = eigenvecs @ np.diag(np.sqrt(eigenvals))
        
        # 構建度量變換矩陣
        Hm = np.eye(4)
        Hm[:3, :3] = np.linalg.inv(L.T)
        
        return Hm
    
    # 1. 估計 IAC
    Omega = estimate_IAC()
    
    # 2. 計算度量變換矩陣
    Hm = compute_metric_transformation(Omega)
    
    # 3. 更新3D點
    points_3d_metric = []
    for X in points_3d_affine:
        X_h = np.append(X, 1)
        X_m = Hm @ X_h
        X_m = X_m[:3] / X_m[3]
        points_3d_metric.append(X_m)
    
    # 4. 更新相機矩陣
    P1_metric = P1_affine @ np.linalg.inv(Hm)
    P2_metric = P2_affine @ np.linalg.inv(Hm)
    
    return np.array(points_3d_metric), P1_metric, P2_metric, Hm

def main():
    # 讀取圖片
    img1 = cv2.imread('hw9/images/homography_images/image_03.jpg')
    img2 = cv2.imread('hw9/images/homography_images/image_02.jpg')
    
    # 選取對應點
    # points1, points2 = select_corresponding_points(img1, img2)
    # print(points1)
    # print("-------------------")
    # print(points2)
    points1 = np.array([[ 751.53645218,  797.73376623],
                        [ 920.39359504,  792.61688312],
                        [1086.69229634,  792.61688312],
                        [1252.99099764,  797.73376623],
                        [ 751.53645218,  966.59090909],
                        [ 930.62736128,  953.7987013 ],
                        [1091.80917946,  946.12337662],
                        [1270.90008855,  956.35714286]])
    
    points2 = np.array([[ 753.74601535,  877.04545455],
                        [ 884.22653483,  877.04545455],
                        [1030.05770366,  877.04545455],
                        [1168.21354782,  877.04545455],
                        [ 756.30445691,  987.05844156],
                        [ 914.92783353,  984.5       ],
                        [1058.2005608 ,  976.82467532],
                        [1214.26549587,  976.82467532]])
  
    # 計算基本矩陣
    F, inliers = compute_fundamental_matrix_with_RANSAC(points1, points2)
    print("Fundamental Matrix:\n", F)
    
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
    points_3d_affine, P1_affine, P2_affine, Ha = upgrade_to_affine(P1, P2, points_3d)
    
    # 計算仿射重建的重投影誤差
    error1_affine = compute_reprojection_error(points_3d_affine, points1, P1_affine)
    error2_affine = compute_reprojection_error(points_3d_affine, points2, P2_affine)
    
    print("\nAffine Reconstruction:")
    print("P1_affine:\n", P1_affine)
    print("P2_affine:\n", P2_affine)
    print(f"Reprojection errors: {error1_affine:.3f}, {error2_affine:.3f}")
    
    # 繪製仿射重建的3D點雲
    plot_3d_points(points_3d_affine, "Affine Reconstruction")
    
    # 3. 度量重建
    points_3d_metric, P1_metric, P2_metric, Hm = upgrade_to_metric(P1_affine, P2_affine, points_3d_affine)
    
    # 計算度量重建的重投影誤差
    error1_metric = compute_reprojection_error(points_3d_metric, points1, P1_metric)
    error2_metric = compute_reprojection_error(points_3d_metric, points2, P2_metric)
    
    print("\nMetric Reconstruction:")
    print("P1_metric:\n", P1_metric)
    print("P2_metric:\n", P2_metric)
    print(f"Reprojection errors: {error1_metric:.3f}, {error2_metric:.3f}")
    
    # 繪製度量重建的3D點雲
    plot_3d_points(points_3d_metric, "Metric Reconstruction")

if __name__ == "__main__":
    main()
