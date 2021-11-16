import os
import sys

fin = open('wider_face_train_bbx_gt.txt','r')
fout = open('wider_face_train_image.txt','a')

while True:
    im_path = fin.readline().replace('\r','').replace('\n','').replace('\t','')
    if not im_path:
        break
    im_path = os.path.splitext(im_path)[0]

    fout.write(im_path+'\n')

    temp = fin.readline().replace('\r','').replace('\n','').replace('\t','')
    num = int(temp)
    while num > 0:
        fin.readline()
        num = num - 1
    pass

fout.close()
fin.close()
