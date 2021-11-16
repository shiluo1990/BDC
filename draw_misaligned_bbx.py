import os
import sys
import cv2
import numpy as np

widerval_results_from = './WIDER_train/'
widerval_results_to = './misaligned_annotations/'
fin = open('wider_face_train_misaligned_bbx.txt','r')

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
        #score = float(annotation[4])
#       add_annotation
        #if score >= thresh:
        cv2.rectangle(im, (x, y), (x+w, y+h),(0, 0, 255), 2, 4, 0)
        num = num - 1
#   store new_result
    if not os.path.exists(os.path.join(widerval_results_to, os.path.dirname(im_path))):
        os.makedirs(os.path.join(widerval_results_to, os.path.dirname(im_path)))
    cv2.imwrite(os.path.join(widerval_results_to, im_path + '.jpg'), im)
    pass
fin.close()
