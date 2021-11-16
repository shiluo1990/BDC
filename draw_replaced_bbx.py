import os
import sys
import cv2
import numpy as np

widerval_results_from = './misaligned_annotations/'
widerval_results_to = './misaligned_detection_results/'
fin = open('wider_face_train_replaced_bbx.txt','r')

while True:
    im_path = fin.readline().replace('\r','').replace('\n','').replace('\t','')
    if not im_path:
        break
    im_path = os.path.splitext(im_path)[0]
    full_im_path = os.path.join(widerval_results_from, im_path + '.jpg')

    temp = fin.readline().replace('\r','').replace('\n','').replace('\t','')
    num = int(temp)

    if num == 0:
        continue

#   read image
    im = cv2.imread(full_im_path)

    while num > 0:
#       read annotation
        annotation = fin.readline().split(" ")
        x = int(annotation[0])
        y = int(annotation[1])
        w = int(annotation[2])
        h = int(annotation[3])
        score = float(annotation[4])
        iou = float(annotation[5])
#       add_annotation
        
        cv2.rectangle(im, (x, y), (x+w, y+h),(0, 255, 0), 2, 4, 0)
        #define font
        font=cv2.FONT_HERSHEY_SIMPLEX
        #image,text,coordinate(up_right),font,size,color,thick
        cv2.putText(im, 'DCS:{:.3f}'.format(score), (x,y-20), font, 0.5, (0, 255, 0), 1)
        cv2.putText(im, 'IoU:{:.3f}'.format(iou), (x,y+h+20), font, 0.5, (0, 255, 0), 1)

        num = num - 1
#   store new_result
    if not os.path.exists(os.path.join(widerval_results_to, os.path.dirname(im_path))):
        os.makedirs(os.path.join(widerval_results_to, os.path.dirname(im_path)))
    cv2.imwrite(os.path.join(widerval_results_to, im_path + '.jpg'), im)
    pass
fin.close()
