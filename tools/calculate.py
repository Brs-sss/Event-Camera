import os
import numpy as np
import cv2

img_w, img_h = 346, 260
means, stdevs = [], []
imgList = np.zeros([img_w, img_h, 3, 1])

base = 'C:/Users/MSI-NB/Desktop/Python Projects/srt/datasets/data_train'
for scene in os.listdir(base):
    for j in [1, 4, 7]:
        path = base + '/' + scene + '/image/000' + str(j) + '.png'
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
            img = cv2.resize(img, (img_h, img_w))

            img = img[:, :, :, np.newaxis]
            imgList = np.concatenate((imgList, img), axis=3)
        except:
            continue

imgList = imgList.astype(np.float32) / 255.

for i in range(3):
    pixels = imgList[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))

# normMean = [0.0030058983, 0.0062538297, 0.01095639]
# normStd = [0.02826636, 0.035462525, 0.040302236]
# transforms.Normalize(normMean = [0.0030058983, 0.0062538297, 0.01095639], normStd = [0.02826636, 0.035462525, 0.040302236])

# normMean = [0.0026769175, 0.0057638837, 0.010363371]
# normStd = [0.027039433, 0.033439837, 0.03856222]
# transforms.Normalize(normMean = [0.0026769175, 0.0057638837, 0.010363371], normStd = [0.027039433, 0.033439837, 0.03856222])
