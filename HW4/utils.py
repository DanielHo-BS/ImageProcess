import cv2
import matplotlib.pyplot as plt
import numpy as np

def gradient_sobel(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(sobel_x, sobel_y, angleInDegrees=True)
    return {'magnitude':magnitude, 'angle':angle}

def drewHist(data,saveName):
    # 繪製柱狀圖)
    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
    plt.figure()
    plt.bar(range(0,len(data_norm)), data_norm, width=1)
    plt.title('Gradient Orientation Histogram')
    plt.xlabel('Angle')
    plt.ylabel('Magnitude')
    plt.savefig(saveName)

def voting(img, magnitude):
    # 設置投票結果陣列
    height, width = img.shape
    votes = np.zeros((360, height, width), dtype=np.uint8)

    # 投票過程
    for ang in range(0, 360):
        radians = ang * np.pi / 180.0
        sin = np.sin(radians)
        cos = np.cos(radians)
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if magnitude[y, x] > 0:
                    x2 = int(x - sin * magnitude[y, x])
                    y2 = int(y + cos * magnitude[y, x])
                    x2 = max(0, min(x2, width - 1))
                    y2 = max(0, min(y2, height - 1))
                    votes[ang, y2, x2] += 1

    return votes


def scoring(hist):
    # 計算分數
    score_list = []
    threshold = np.mean(hist) - np.std(hist)
    for hist_angle in range(0, 360):
        score = 0
        if hist[hist_angle] > threshold:
            for ang in range(0, 360):
                ang1 = hist_angle-ang if hist_angle-ang >= 0 else hist_angle-ang+360
                ang2 =  hist_angle+ang if hist_angle+ang <= 359 else hist_angle+ang-360
                score = score + hist[ang1] * hist[ang2]
        score_list.append(score)


    # 找出最好的分數的角度
    candidates_list = []
    for ang in range(0, 360):
        if score_list[ang] == np.max(score_list):
            best_angle = ang
            candidates_list.append(ang)
    print("Candidate Angles: ",candidates_list)

    return {'score':score_list, 'candidates':candidates_list, 'best_angle':best_angle}  

def centerPoint(img, votes, angle):
    height, width = img.shape
    # 找到最佳投票結果中的中點
    points = []

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if votes[angle, y, x] > 0:
                points.append((x, y))
    
    points = np.array(points)
    x = np.mean(points[:, 0])
    y = np.mean(points[:, 1])
    return [x, y]

def symmetryAxis(img, angle, point, saveName):
    x, y = point[0], point[1] 
    # 計算對稱軸方程式
    radians = angle * np.pi / 180.0
    sin = np.sin(radians)
    cos = np.cos(radians)
    pt1 = (int(x - 1000 * sin), int(y + 1000 * cos))
    pt2 = (int(x + 1000 * sin), int(y - 1000 * cos))
    # 繪製對稱軸
    img_symmetry = cv2.line(img, pt1, pt2, (0, 0, 255), 2)
    plt.figure()
    plt.imshow(cv2.cvtColor(img_symmetry, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig(saveName)
    return img_symmetry

