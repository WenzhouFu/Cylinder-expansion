# -*- coding:utf-8 -*-
# organization : 图像识别
# @Author      : Wen'zhou Fu
# @Email       : WenzhouFu@tom.com
# @Time        : 2020/12/15 11:27

from matplotlib.pyplot import plot, show
import numpy as np
import cv2


class Parameter:
    def __init__(self, r, d, path_name, ):
        self.path_name = path_name  # 图像路径
        self.r = r  # 物体半径
        self.d = d  # 倍率px/cm
        self.image = cv2.imread(self.path_name)
        self.height, self.width = self.image.shape[:2]

    @staticmethod
    def chord_length(r, x):  # 计算投影面弦长
        length = r ** 2 - x ** 2
        return length if length >= 0 else 0

    @staticmethod
    def function_image(x, nb):
        plot(x, nb)
        show()

    # x轴偏移量

    def x_axis_offset(self, x):
        center_x = x.mean()
        r = self.r * self.d

        x = x - center_x  # 各像素点与图像中心的差

        n = r ** 2 - x ** 2  # 图像中校正半径计算（以中心为基准）
        n[n < 0] = 0  # 进行缺省附加
        y = -np.sqrt(n) + r  # 点高度
        y[n == 0] = 0
        z = x**2+y**2  # 弦长

        # self.function_image(x, z)
        cos_a = 1 - z / (2 * r ** 2)  # 三角形公式 cosa = （r^2+r^2-z^2）/2*r^2
        cos_a[cos_a < -1] = 0  # 去除缺省值
        cos_a = np.arccos(cos_a)  # 圆心角
        arc = cos_a * 3.14 * r / 20  # 弧长
        arc[x < 0] = -arc[x < 0]  # 以图形中心为原点进行偏移

        m = np.expand_dims(arc, 0).repeat(self.height, axis=0).reshape(-1)  # 维度改变
        return m

    @staticmethod
    def save_array_as_npy(fname, array):
        np.save(fname, array)

    @staticmethod
    def load_array_as_npy(fname, array):
        return np.load(fname, array)

    def get_mapping_relation(self, a, b, c, d, name):
        x, y = np.meshgrid(range(self.width), range(self.height))
        t = self.x_axis_offset(x[0])
        x = x.reshape(-1)
        y = y.reshape(-1)
        location_of_source_image = np.stack([x, y], 1)
        center_x = x.mean()
        center_y = y.mean()
        center = np.array([center_x, center_y])
        norm = np.mean(center)
        dist = np.sqrt(((location_of_source_image - center) ** 2).sum(1))
        r = np.sqrt(((x - center_x) / norm) ** 2 + ((y - center_y) / norm) ** 2)
        rdest = (a * r ** 4 + b * r ** 3 + c * r ** 2 + d * r) * norm
        target_x = x - t
        target_y = rdest / dist * (y - center_y) + center_y
        location_of_dest_image = np.stack([target_x, target_y], 1)
        # self.save_array_as_npy(name, location_of_dest_image)
        return location_of_dest_image

if __name__ == '__main__':
    radius = 6.285
    magnification = 115
    mapping_where_to_save = "./location_of_dest_image.npy"
    param_a = 0.1  # 只影响图像的最外层像素
    param_b = -0.3  # 大多数情况下只需要b优化
    param_c = 0.04  # 最均匀校正
    param_d = 1 - param_a - param_b - param_c  # 描述了图像的线性缩放
    original_image_name = "./1.bmp"
    sa = Parameter(radius, magnification, original_image_name)
    dest_array = sa.get_mapping_relation(param_a, param_b, param_c, param_d, mapping_where_to_save)
    dest_array = dest_array.reshape((-1, 2))
    color = cv2.imread(original_image_name)
    color = cv2.resize(color, tuple(color.shape[:2][::-1]))
    width, height = color.shape[1], color.shape[0]
    map_x = dest_array[:, 1].reshape((height, width)).astype(np.float32)
    map_y = dest_array[:, 0].reshape((height, width)).astype(np.float32)

    mapped_img = cv2.remap(color, map_y, map_x, cv2.INTER_LINEAR)  # 重映射

    cv2.imwrite("./5554.bmp", mapped_img)


















