import cv2


''' Script to split the simulator images because they had 2 side view
'''


img = "prova2.jpg"
img = cv2.imread(img)

img2 = img[ :, 0 : 2560//2]
# print(img2.shape)
cv2.imshow("Original", img)

cv2.imwrite("C:\\Users\\koki\\Desktop\\prova2.jpg", img2)

key=cv2.waitKey(0)
if (key==27):
    cv2.destroyAllWindows()
