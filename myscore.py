from sklearn.externals import joblib
import numpy as np


def binary_pic(num_img: np.ndarray):
    # 图片二值话只有0,1
    select_num_index = np.abs(num_img) >= 0.2
    assert isinstance(select_num_index, np.ndarray)
    return select_num_index.astype(int)


def num_area(img):
    # 截取得分所在区域, 即上下左右没有空白区域

    def col(num_img):
        col_sum = np.sum(num_img, axis=0)
        length = col_sum.shape[0]
        start = 0
        end = 0
        for len_i in range(length):
            if col_sum[len_i] > 0:
                start = len_i
                break

        for len_j in range(length-1, 0, -1):
            if col_sum[len_j] > 0:
                end = len_j
                break
        return start, end+1

    i, j = col(img)
    row_sum = np.sum(img, axis=1)
    new_img = img[row_sum != 0, i:j]
    return new_img


def split_num(num_img):
    # 将图片上的文字切分出来,例如123,切分成1, 2, 3便于预测
    col_sum = np.sum(num_img, axis=0)
    sum_index = col_sum > 0
    assert isinstance(sum_index, np.ndarray)

    def value_change(l):
        select_index = [0, ]
        for i in range(1, len(l)):
            if l[i-1] != l[i]:
                select_index.append(i)
        select_index.append(len(l))
        return select_index

    split_img = list()
    split_slice = value_change(sum_index)
    for img_i in range(0, len(split_slice) // 2):
        start_i, end_i = split_slice[img_i * 2: img_i * 2 + 2]
        split_img.append(num_img[:, start_i: end_i])
    return split_img


def chain(img):
    img = binary_pic(img)
    img = num_area(img)
    img = split_num(img)
    # io.imshow(img)
    return img


def read_in_img(img):
    # 从1080*1920的图片上将分数大概范围截取出来
    part_img = img[200:300, 50:500]
    out_img = chain(part_img)
    return out_img


def flat_img(pic_list):
    # 将一张图片统一改成1 * 6300的向量
    data = np.zeros((len(pic_list), 6300), dtype=int)
    for i, split_part_pic in enumerate(pic_list):
        # sample_data = np.zeros((1, 6300)).ravel()
        # sample_data[-1] = int(target[sample_i])
        pic_len = split_part_pic.shape[0] * split_part_pic.shape[1]
        try:
            data[i, 0:pic_len] = split_part_pic.ravel()
        except ValueError:
            print(split_part_pic.shape)
    return data


def predict(img):
    split_im = read_in_img(img)
    predict_data = flat_img(split_im)
    predict_value = lr.predict(predict_data)
    s = ""
    for i in predict_value:
        s += str(i)
    return int(s)


# 已经训练好的模型
lr = joblib.load("logistic.model")
