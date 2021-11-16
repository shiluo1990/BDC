import os
import sys

widerval_pred = './prediction/'
widerval_formulation = './formulation/'
fin = open('wider_face_train_image.txt','r')

while True:
    im_path = fin.readline().replace('\r','').replace('\n','').replace('\t','')
    if not im_path:
        break
    im_path = os.path.splitext(im_path)[0]
    full_im_predict_path = os.path.join(widerval_pred, im_path + '.txt')
    full_im_formulation_path = os.path.join(widerval_formulation, im_path + '.txt')

#   mkdir
    if not os.path.exists(os.path.join(widerval_formulation, os.path.dirname(im_path))):
        os.makedirs(os.path.join(widerval_formulation, os.path.dirname(im_path)))

#   open prediction txt
    fdet = open(full_im_predict_path,'r')
    fout = open(full_im_formulation_path,'a')

    fout.write(im_path+'.jpg')
    fout.write('\n')
    fdet.readline()    

    temp = fdet.readline().replace('\r','').replace('\n','').replace('\t','')
    num = int(temp)
    
    if num >= 0:
        fout.write("%d\n" % num)

    while num > 0:
        fout.write(fdet.readline())
        num = num - 1

    fdet.close()
    fout.close()

    pass

fin.close()
