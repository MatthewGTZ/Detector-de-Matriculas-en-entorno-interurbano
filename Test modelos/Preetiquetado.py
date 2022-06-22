import cv2
import os
input_images_path = "D:/Escritorio/tfg/capturas/input"
files_names = os.listdir(input_images_path)
print(files_names)
output_images_path = "D:/Escritorio/tfg/capturas/output"
if not os.path.exists(output_images_path):
    os.makedirs(output_images_path)
    print("Directorio creado: ", output_images_path)
count = 0
for file_name in files_names:
    #print(file_name)
    '''
    if file_name.split(".")[-1] not in ["jpeg", "png"]:
        continue
    '''
    image_path = input_images_path + "/" + file_name
    print(image_path)
    image = cv2.imread(image_path)
    if image is None:
        continue
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(output_images_path + "/image" + str(count) + ".jpg", image)
    count += 1
    '''
    cv2.imshow("Image", image)
    cv2.waitKey(0)
cv2.destroyAllWindows()
'''