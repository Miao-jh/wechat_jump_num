import os
import random
import subprocess
import time
from io import BytesIO, StringIO

import numpy as np
from skimage import img_as_ubyte, io, feature, color, filters
import cv2
from myscore import predict


def get_screenshot():
    subprocess.Popen("adb devices", shell=True, stdout=subprocess.PIPE)
    process = subprocess.Popen(
        'adb shell screencap -p',
        shell=True, stdout=subprocess.PIPE)
    binary_pic = process.stdout.read()
    binary_pic = binary_pic.replace(b'\r\r\n', b'\n')
    try:
        image = io.imread(BytesIO(binary_pic), as_grey=True)
    except (SyntaxError, ValueError, TypeError):
        os.system('adb shell screencap -p /sdcard/autojump.png')
        os.system('adb pull /sdcard/autojump.png .')
        image = io.imread('./autojump.png', as_grey=True)
    return image


def jump(distance):
    # 这个参数还需要针对屏幕分辨率进行优化
    press_time = int(distance * 1.36) + int(random.uniform(40, 50))
    press_time = max([press_time, 250])

    # 生成随机手机屏幕模拟触摸点
    # 模拟触摸点如果每次都是同一位置，成绩上传可能无法通过验证
    rand1 = random.uniform(10, 160)
    rand2 = random.uniform(-30, 100)
    rand3 = random.uniform(10, 100)
    rand4 = random.uniform(-50, 100)
    cmd = ('adb shell input swipe %i %i %i %i ' + str(press_time)) \
          % (320 + rand1, 410 + rand2, 320 + rand3, 410 + rand4)
    os.system(cmd)
    print(cmd)


def get_center(img_canny, ):
    # 利用边缘检测的结果寻找物块的上沿和下沿
    # 进而计算物块的中心点
    y_sum = np.sum(img_canny[300:1500], axis=1)
    y_sum[y_sum >= 1] = 1
    y_top = np.argmax(y_sum) + 300
    x_top = int(np.mean(np.nonzero(canny_img[y_top])))

    # 从上倒下扫描
    y_bottom = y_top + 50
    for row in range(y_bottom, H):
        if canny_img[row, x_top] != 0:
            y_bottom = row
            break

    x_center, y_center = x_top, (y_top + y_bottom - 20) // 2
    return img_canny, x_center, y_center


def img_minus(img):
    # 之前版本的游戏颜色是从手机上到下的渐变,将列的结果减去第一列可以得到去除背景的效果,提高边缘检测的准确性
    img_grey2 = img - img[:, 0].reshape((img.shape[0], 1))
    img_grey2 = np.abs(img_grey2)
    add_index = (img_grey2 > 0.01)  # 去除部门噪声点
    img_grey2[add_index] = img_grey2[add_index] * 2  # 加大正常值
    img_grey2[add_index] = img_grey2[add_index] + 0.2
    ct_index = img_grey2 > 1
    img_grey2[ct_index] = 1
    return img_grey2



# 匹配小跳棋的模板
temp1 = io.imread('temp_player.jpg', as_grey=True)
w1, h1 = temp1.shape[::-1]
# 匹配游戏结束画面的模板
temp_end = io.imread('temp_end.jpg', as_grey=True)

score = 10000

# 循环直到游戏失败结束
for i in range(10000):
    times = 0
    while True:
        times += 1
        img_grey = get_screenshot()
        img_grey2 = img_minus(img_grey)
        predict_score = predict(img_grey2)  # 识别得分
        if predict_score > score or predict_score == 0 or times > 3:
            # 如果分数提高,说明小球已经跳到位置,开始进行下一步截图等
            break
        time.sleep(0.3)
    print("this score is: ", predict_score)
    # 如果在游戏截图中匹配到带"再玩一局"字样的模板，则循环中止
    if times > 3:
        res_end = feature.match_template(img_grey, temp_end)
        if res_end.max() > 0.95:
            print('Game over!')
            break

    # 模板匹配截图中小跳棋的位置
    res1 = feature.match_template(img_grey, temp1)
    pos = np.unravel_index(np.argmax(res1), res1.shape)
    max_loc1 = pos[-1], pos[0]
    center1_loc = (max_loc1[0] + 39, max_loc1[1] + 189)

    canny_img = feature.canny(img_grey2, sigma=1)
    # io.imshow(canny_img)
    # sobel_img = filters.sobel_h(img_rgb)
    H, W = canny_img.shape

    # 消去小跳棋轮廓对边缘检测结果的干扰，同时防止截图是出现超过某个好友或者其他特效的影响
    canny_img[max_loc1[1] - 189: max_loc1[1] + 300, max_loc1[0] - 30: max_loc1[0] + 107] = 0
    img_grey, x_center, y_center = get_center(canny_img)

    # 将图片输出以供调试
    img_grey = cv2.circle(img_as_ubyte(img_grey), (x_center, y_center), 10, 255, -1)
    img_grey = cv2.circle(img_as_ubyte(img_grey), center1_loc, 10, 255, -1)
    # cv2.rectangle(img_as_ubyte(canny_img), max_loc1, center1_loc, 255, 2)
    cv2.imwrite('last.png', img_grey)

    distance = (center1_loc[0] - x_center) ** 2 + (center1_loc[1] - y_center) ** 2
    distance = distance ** 0.5
    print(distance)
    jump(distance)
    time.sleep(random.uniform(0.5, 1.5))
