import numpy as np
import matplotlib.pyplot as plt
import cv2

def compute_fundamental_matrix(points1, points2):
    """
    計算基本矩陣 F
    points1, points2: 兩張圖片中對應的特徵點 (n×2 numpy array)
    """
    # 1. 資料標準化
    def normalize_points(points):
        # 計算質心
        centroid = np.mean(points, axis=0)
        # 計算到質心的平均距離
        dist = np.sqrt(np.sum((points - centroid) ** 2, axis=1))
        scale = np.sqrt(2) / np.mean(dist)
        
        # 標準化矩陣
        T = np.array([
            [scale, 0, -scale*centroid[0]],
            [0, scale, -scale*centroid[1]],
            [0, 0, 1]
        ])
        
        # 轉換為齊次座標
        points_h = np.hstack((points, np.ones((points.shape[0], 1))))
        # 應用標準化
        normalized_points = (T @ points_h.T).T
        
        return normalized_points[:, :2], T

    # 標準化兩組點
    norm_points1, T1 = normalize_points(points1)
    norm_points2, T2 = normalize_points(points2)

    # 2. 建立方程組
    n = len(points1)
    A = np.zeros((n, 9))
    for i in range(n):
        x1, y1 = norm_points1[i]
        x2, y2 = norm_points2[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

    # 3. 使用SVD求解
    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # 4. 強制F的秩��2
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0  # 將最小的奇異值設為0
    F = U @ np.diag(S) @ Vt

    # 5. 反標準化
    F = T2.T @ F @ T1

    # 6. 正規化F
    F = F / F[2,2]
    return F

def compute_fundamental_matrix_with_RANSAC(points1, points2, threshold=1.0, num_iterations=1000):
    """
    使用RANSAC算法計算基本矩陣
    """
    best_F = None
    best_inliers = []
    max_inliers = 0

    for _ in range(num_iterations):
        # 1. 隨機選擇8個點
        idx = np.random.choice(len(points1), 8, replace=False)
        sample_points1 = points1[idx]
        sample_points2 = points2[idx]

        # 2. 計算F
        F = compute_fundamental_matrix(sample_points1, sample_points2)

        # 3. 計算所有點的誤差
        inliers = []
        for i in range(len(points1)):
            p1 = np.append(points1[i], 1)
            p2 = np.append(points2[i], 1)
            
            # 計算極線距離
            line = F @ p1
            error = abs(np.dot(p2, line)) / np.sqrt(line[0]**2 + line[1]**2)
            
            if error < threshold:
                inliers.append(i)

        # 4. 更新最佳結果
        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            best_inliers = inliers
            best_F = F
        


    # 5. 使用���有內點重新計算F
    if best_inliers:
        best_F = compute_fundamental_matrix(
            points1[best_inliers],
            points2[best_inliers]
        )

    return best_F, best_inliers

def select_corresponding_points(img1, img2, num_points=8):
    """
    使用matplotlib手動選取兩張圖片中的對應特徵點
    """
    points1 = []
    points2 = []
    
    # 創建並排顯示的圖片
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # 顯示圖片
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    
    ax1.set_title(f'Image 1 - Selected 0/{num_points} points')
    ax2.set_title(f'Image 2 - Selected 0/{num_points} points')
    
    def onclick(event):
        # 確保還沒選完所有點
        if len(points2) < num_points:  # 修改判斷條件
            if event.inaxes == ax1 and len(points1) <= len(points2):
                points1.append([event.xdata, event.ydata])
                ax1.plot(event.xdata, event.ydata, 'ro')
                ax1.text(event.xdata+5, event.ydata+5, str(len(points1)), color='red')
                plt.draw()
                
                # 更新標題顯示進度
                ax1.set_title(f'Image 1 - Selected {len(points1)}/{num_points} points')
                
            elif event.inaxes == ax2 and len(points2) < len(points1):
                points2.append([event.xdata, event.ydata])
                ax2.plot(event.xdata, event.ydata, 'ro')
                ax2.text(event.xdata+5, event.ydata+5, str(len(points2)), color='red')
                plt.draw()
                
                # 更新標題顯示進度
                ax2.set_title(f'Image 2 - Selected {len(points2)}/{num_points} points')
                
                # 如果已經選取足夠的點，更新標題
                if len(points2) == num_points:
                    ax1.set_title('All points selected - Close window to continue')
                    ax2.set_title('All points selected - Close window to continue')
    
    # 綁定點擊事件
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    # 確保選取了足夠的點
    if len(points1) != num_points or len(points2) != num_points:
        print(f"Warning: Expected {num_points} points but got {len(points1)} and {len(points2)} points")
        return None, None
    
    # 轉換為numpy數組
    return np.array(points1), np.array(points2)

def main():
    # 讀取兩張圖片
    img1 = cv2.imread('hw9/images/homography_images/171084.jpg')
    img2 = cv2.imread('hw9/images/homography_images/171085.jpg')
    
    if img1 is None or img2 is None:
        print("無法讀取圖片！")
        return
    
    # 選取對應點
    points1, points2 = select_corresponding_points(img1, img2, num_points=8)
    
    if points1 is None or points2 is None:
        print("點選取失敗！")
        return
    
    # 計算基本矩陣
    F, inliers = compute_fundamental_matrix_with_RANSAC(points1, points2)
    
    print("Fundamental Matrix:")
    print(F)
    print(f"Number of inliers: {len(inliers)}")
    
    # 視覺化結果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    
    # 繪製所有點和對應關係
    for i in range(len(points1)):
        color = 'g' if i in inliers else 'r'
        ax1.plot(points1[i,0], points1[i,1], color+'o')
        ax2.plot(points2[i,0], points2[i,1], color+'o')
        ax1.text(points1[i,0]+5, points1[i,1]+5, str(i+1), color=color)
        ax2.text(points2[i,0]+5, points2[i,1]+5, str(i+1), color=color)
        
    ax1.set_title('Image 1 with feature points')
    ax2.set_title('Image 2 with feature points')
    plt.show()

if __name__ == "__main__":
    main()
