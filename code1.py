import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.color import rgb2gray, label2rgb
from skimage.filters import sobel
from skimage import io
from skimage.segmentation import watershed
import scipy.ndimage as nd
img = io.imread(r"/content/Screenshot 2024-08-12 213540.png")
img = img[:, :, :3] if img.shape[-1] == 4 else img
gray_img = rgb2gray(img)
map = sobel(gray_img)
mark = np.zeros_like(gray_img, dtype=int)
mark[gray_img < 0.2] = 1
mark[gray_img > 0.6] = 2
seg = watershed(map, mark)
seg_fill = nd.binary_fill_holes(seg - 1)
lab, _ = nd.label(seg_fill)
mask = np.zeros(gray_img.shape)
mask[25:-25, 25:-25] = 1
# Plot results
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1).imshow(img), plt.title('Original Image'), plt.axis('off')
plt.subplot(2, 2, 2).imshow(mask, cmap='gray'), plt.title('Initial Contour Location'), plt.axis('off')
plt.subplot(2, 2, 3).imshow(io.imread(r"/content/Screenshot 2024-08-12 224615.png"), cmap='gray'), plt.title('Segmented image,100 iterations'), plt.axis('off')
plt.subplot(2, 2, 4).imshow(seg, cmap='nipy_spectral'), plt.title("Segmented image,300 iterations"), plt.axis('off')
plt.tight_layout()
plt.show()
