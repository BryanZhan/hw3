import cv2 as cv
import numpy as np

img = cv.imread("mp2a.jpg")

g = img[:, :, 0]
b = img[:, :, 1]
r = img[:, :, 2]

hist_r, bins_r = np.histogram(r, 256)
hist_g, bins_g = np.histogram(g, 256)
hist_b, bins_b = np.histogram(b, 256)

cdf_r = hist_r.cumsum()
cdf_g = hist_g.cumsum()
cdf_b = hist_b.cumsum()

cdf_r = (cdf_r - cdf_r[0]) * 255 / (cdf_r[-1] - 1)
cdf_r = cdf_r.astype(np.uint8)
cdf_g = (cdf_g - cdf_g[0]) * 255 / (cdf_g[-1] - 1)
cdf_g = cdf_g.astype(np.uint8)
cdf_b = (cdf_b - cdf_b[0]) * 255 / (cdf_b[-1] - 1)
cdf_b = cdf_b.astype(np.uint8)

r2 = cdf_r[r]
g2 = cdf_g[g]
b2 = cdf_b[b]

img2 = img.copy()
img2[:, :, 0] = g2
img2[:, :, 1] = b2
img2[:, :, 2] = r2

cv.imshow("img2", img2)

img3 = cv.imread("mp2a.jpg", 1)
(b, g, r) = cv.split(img3)
bH = cv.equalizeHist(b)
gH = cv.equalizeHist(g)
rH = cv.equalizeHist(r)
result = cv.merge((bH, gH, rH))
cv.imshow("opencv", result)
cv.waitKey(0)
