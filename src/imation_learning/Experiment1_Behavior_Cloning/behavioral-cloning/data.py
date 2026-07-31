import numpy as np
import skimage.transform as sktransform
import random
import matplotlib.image as mpimg
import os


# 使用三个摄像头
cameras = ['left', 'center', 'right']

# 左右摄像头角度修正
cameras_steering_correction = [0.25, 0.0, -0.25]


def preprocess(image, top_offset=0.375, bottom_offset=0.125):
    """
    图像预处理:
    1. 裁剪天空和车头部分
    2. resize到CNN输入大小
    3. 归一化
    """

    top = int(top_offset * image.shape[0])
    bottom = int(bottom_offset * image.shape[0])

    image = image[top:-bottom, :]

    image = sktransform.resize(
        image,
        (32, 128, 3)
    )

    return image.astype(np.float32)



def generate_samples(data, root_path, augment=True):
    """
    Keras Generator

    输出:
        X:
        (batch,32,128,3)

        y:
        steering angle
    """

    while True:

        indices = np.random.permutation(len(data))

        batch_size = 128


        for batch_start in range(0, len(indices), batch_size):

            batch_indices = indices[
                batch_start:
                batch_start + batch_size
            ]


            x = []
            y = []


            for index in batch_indices:

                # 随机选择摄像头
                if augment:
                    camera = np.random.randint(3)
                else:
                    camera = 1


                img_path = os.path.join(
                    root_path,
                    data[cameras[camera]].iloc[index].strip()
                )


                image = mpimg.imread(img_path).astype(np.float32)


                angle = (
                    data['steering'].iloc[index]
                    +
                    cameras_steering_correction[camera]
                )


                # 数据增强：随机阴影
                if augment:

                    h, w = image.shape[:2]

                    x1, x2 = np.random.choice(
                        w,
                        2,
                        replace=False
                    )


                    if x1 > x2:
                        x1, x2 = x2, x1


                    shadow = np.random.randint(
                        0,
                        3
                    )


                    if shadow:

                        image[:, x1:x2, :] *= 0.5


                # 随机裁剪变化

                if augment:

                    top = random.uniform(
                        0.325,
                        0.425
                    )

                    bottom = random.uniform(
                        0.075,
                        0.175
                    )

                else:

                    top = 0.375
                    bottom = 0.125



                image = preprocess(
                    image,
                    top,
                    bottom
                )


                x.append(image)
                y.append(angle)



            x = np.array(x)
            y = np.array(y)


            # 随机水平翻转

            if augment:

                flip_indices = random.sample(
                    range(len(x)),
                    len(x)//2
                )


                x[flip_indices] = (
                    x[flip_indices,:,::-1,:]
                )


                y[flip_indices] = (
                    -y[flip_indices]
                )


            yield x, y