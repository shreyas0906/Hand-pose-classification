import cv2 
import numpy as np 

image = np.zeros((640,480,3))
point_1 = np.array((100,100))
point_2 = np.array((300,100))
point_3 = np.array((400, 400))

line = [point_1, point_2]

v1 = [line[1][0] - line[0][0], line[1][1] - line[0][1]]
v2 = [line[1][0] - point_3[0], line[1][1] - point_3[1]]

xp = (v1[0] * v2[1]) - (v1[1] * v2[0])

print(xp)

cv2.line(image, point_1, point_2, (0,255,0), 2)
cv2.circle(image, point_3, radius=0, color=(0, 0, 255), thickness=5)
cv2.imshow('test', image)

cv2.waitKey(0)
cv2.destroyAllWindows()