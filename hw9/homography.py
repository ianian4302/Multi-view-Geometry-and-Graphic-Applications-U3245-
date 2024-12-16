import cv2
import numpy as np
import matplotlib.pyplot as plt

def onclick(event):
    global pts_count, current_pts
    if event.button == 1:  # 左鍵點擊
        if pts_count < 4:
            current_pts[pts_count] = [event.xdata, event.ydata]
            plt.plot(event.xdata, event.ydata, 'ro')
            plt.draw()
            pts_count += 1
            print(f'點 {pts_count}: ({event.xdata:.0f}, {event.ydata:.0f})')

if __name__ == '__main__':
    # 讀取源圖像
    im_src = cv2.imread('hw9/images/homography_images/image_03.jpg')
    im_src_rgb = cv2.cvtColor(im_src, cv2.COLOR_BGR2RGB)
    
    # 初始化變數
    pts_count = 0
    pts_src = np.zeros((4, 2))
    current_pts = pts_src  # 指向當前正在使用的點陣列
    
    # 使用 matplotlib 顯示圖片並綁定點擊事件
    # fig = plt.figure()
    # plt.imshow(im_src_rgb)
    # plt.title('請點選4個角點 (順時針方向)')
    # fig.canvas.mpl_connect('button_press_event', onclick)
    # plt.show()
    
    # 讀取目標圖像
    im_dst = cv2.imread('hw9/images/homography_images/image_02.jpg')
    im_dst_rgb = cv2.cvtColor(im_dst, cv2.COLOR_BGR2RGB)
    
    # 重置計數器用於目標圖像
    pts_count = 0
    pts_dst = np.zeros((4, 2))
    current_pts = pts_dst  # 切換到目標點陣列
    
    # 為目標圖像重複相同過程
    # fig = plt.figure()
    # plt.imshow(im_dst_rgb)
    # plt.title('請點選4個角點 (與源圖像相對應的位置)')
    # fig.canvas.mpl_connect('button_press_event', onclick)
    # plt.show()
    
    # 將點轉換為正確的格式
    pts_src = np.float32(pts_src)
    pts_dst = np.float32(pts_dst)
    # 預定義的點座標
    pts_src = np.float32([
        [744, 645],  # 點1
        [1232, 647], # 點2
        [1282, 1114],# 點3
        [756, 1121]  # 點4
    ])

    pts_dst = np.float32([
        [741, 787],  # 點1
        [1156, 785], # 點2
        [1239, 1090],# 點3
        [765, 1100]  # 點4
    ])
    
    # 計算單應性矩陣
    h, status = cv2.findHomography(pts_src, pts_dst)
    
    # 根據單應性矩陣對源圖像進行變換
    im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
    
    # 顯示結果
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(im_src_rgb)
    plt.title('Source Image')
    
    plt.subplot(132)
    plt.imshow(im_dst_rgb)
    plt.title('Destination Image')
    
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB))
    plt.title('Warped Source Image')
    
    plt.show()