import argparse
import cv2
import matplotlib
import numpy as np
import os
from utils import gradient_sobel, drewHist
from utils import voting, scoring, centerPoint, symmetryAxis

def main(inputFile, outputPath):
    # 讀取圖像
    img = cv2.imread(inputFile)
    # 轉為灰度圖像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 計算梯度和方向
    gradient = gradient_sobel(gray)

    # 繪製梯度方向柱狀圖
    hist, _ = np.histogram(gradient['angle'], bins=360, range=(0, 360))
    saveName = outputPath + '/histogram_angle.jpg'
    drewHist(hist, saveName)

    # 投票
    votes = voting(gray, gradient['magnitude'])
    votes_hist = [np.sum(votes[ang]) for ang in range(0, 360)]
    # 繪製投票柱狀圖
    saveName = outputPath + '/histogram_vote.jpg'
    drewHist(votes_hist, saveName)

    # 分數篩選
    result = scoring(votes_hist)
    # 繪製分數柱狀圖
    saveName = outputPath + '/histogram_score.jpg'
    drewHist(result['score'], saveName)

    # 找到中點
    point = centerPoint(gray, votes, result['best_angle'])
    print('Center Point: ', point)

    # 繪製對稱軸並顯示結果
    saveName = outputPath + '/output.jpg'
    img_symmetry = symmetryAxis(img, result['best_angle'], point, saveName)
    



if __name__ == '__main__':
    
    matplotlib.use('Agg')  
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./Input/image.jpg', help='the file path of input: *.jpg or *.png')
    parser.add_argument('--output', type=str, default='./Output', help='save image(s) path')
    opt = parser.parse_args()
    
    if not os.path.exists(opt.output):
        os.mkdir(opt.output)

    main(opt.input, opt.output)

