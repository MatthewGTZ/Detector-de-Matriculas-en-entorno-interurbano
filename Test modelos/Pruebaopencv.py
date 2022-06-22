import cv2

desired_size = 368
im_pth = "D:/Escritorio/Prueba/Imagenes/18148570.jpg"

im = cv2.imread(im_pth)
old_size = im.shape[:2] # old_size is in (height, width) format

ratio = float(desired_size)/max(old_size)
new_size = tuple([int(x*ratio) for x in old_size])

# new_size should be in (width, height) format

im = cv2.resize(im, (new_size[1], new_size[0]))

delta_w = desired_size - new_size[1]
delta_h = desired_size - new_size[0]
top, bottom = delta_h//2, delta_h-(delta_h//2)
left, right = delta_w//2, delta_w-(delta_w//2)

color = [0, 0, 0]
new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
    value=color)

gris = cv2.cvtColor(new_im, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gris, 50, 150)

cv2.imshow("image", canny)
cv2.waitKey(0)
cv2.destroyAllWindows()



