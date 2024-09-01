import cv2
import numpy as np
from PIL import Image

img = np.array(Image.open('/media/jslee/Data2/jslee_two/jisu/VPD/depth/dataset/nyu_depth_v2/label/label_train/label980.png'))
# np.savetxt('./text_exam.txt', img)
print(img) # 0 1 2

# cv2.imread 0, 128, 255