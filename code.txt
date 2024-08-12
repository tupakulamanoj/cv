import cv2
import numpy as np
from google.colab.patches import cv2_imshow

image = cv2.imread(r"/content/Screenshot 2024-08-12 213540.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 150, 255)
ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
cv2_imshow(image)
cv2_imshow(contour_image)
cv2_imshow(thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
