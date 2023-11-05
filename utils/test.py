from diffusers.utils import load_image
from PIL import Image
import cv2
import numpy as np
from diffusers.utils import load_image

# canny_image = load_image(
#     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/landscape.png"
# )
# canny_image = np.array(canny_image)
# low_threshold = 100
# high_threshold = 200

# canny_image = cv2.Canny(canny_image, low_threshold, high_threshold)

# # zero out middle columns of image where pose will be overlayed
# zero_start = canny_image.shape[1] // 4
# zero_end = zero_start + canny_image.shape[1] // 2
# canny_image[:, zero_start:zero_end] = 0

(512,512,3).shape[-2:]