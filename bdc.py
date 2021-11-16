# Bounding-Box Deep Calibration
# Input:
# 1)Original annotations: wider_face_train_bbx_gt.txt
# 2)Detection results on WIDER FACE training set: ./formulation/
# Output:
# Calibrated annotations: wider_face_train_bbx_gt_new.txt
import os
import sys
import numpy as np

from iou import bbox_iou

# Original annotations
fin = open('wider_face_train_bbx_gt.txt','r')
# Location of detection results on WIDER FACE training set
tinaface_train_predict = './formulation/'
# Calibrated annotations
fout = open('wider_face_train_bbx_gt_calibrated.txt','w')

# Misaligned annotations
fout_before = open('wider_face_train_misaligned_bbx.txt','w')
# Replaced bounding-boxes
fout_after = open('wider_face_train_replaced_bbx.txt','w')

ADC = 0.568973	# ADC: Average Detection Confidence,  0.568973
Tm = 0.5    # Matching iou threshold
Tc = 0.8    # Calibrating iou threshold

image_num = 0

total_calibrated = 0

while True:
    im_path = fin.readline().replace('\r','').replace('\n','').replace('\t','')
    if not im_path:
        break

    image_num = image_num + 1
    print("Processing image %d" % image_num)

    fout.write(im_path+'\n')
    fout_before.write(im_path+'\n')
    fout_after.write(im_path+'\n')

#   construct the path of predicted detection results
    im_path = os.path.splitext(im_path)[0]
    predict_anno_path = os.path.join(tinaface_train_predict, im_path + '.txt')

#   open predicted detection results files
    fin_predict = open(predict_anno_path,'r')

########################################################################################################################
#   Obtain original annotations
    gt_anno = []
    gt_bbox = []

    gt_num = int(fin.readline().replace('\r','').replace('\n','').replace('\t',''))

    # current training image has no annotated bbox
    if gt_num == 0:
        break

    # write the annotation number of current image, and construct calibrated annotations
    fout.write(str(gt_num)+'\n')

    gt_index = gt_num

    while gt_index > 0:
        gt_anno_line = fin.readline().split(" ")
        x = int(gt_anno_line[0])
        y = int(gt_anno_line[1])
        w = int(gt_anno_line[2])
        h = int(gt_anno_line[3])
        blur = int(gt_anno_line[4])
        expression = int(gt_anno_line[5])
        illumination = int(gt_anno_line[6])
        invalid = int(gt_anno_line[7])
        occlusion = int(gt_anno_line[8])
        pose = int(gt_anno_line[9])

        # store original annotations
        gt_anno.append([x,y,w,h,blur,expression,illumination,invalid,occlusion,pose])
        # store original bounding-boxes, [x_min,y_min,x_max,y_max]
        gt_bbox.append([x,y,x+w,y+h])

        gt_index = gt_index - 1

########################################################################################################################
#   Obtain predicted detection results
#   Only store high confidence detection results (HCDR), whose detection confidence score is greater than ADC
    # store predicted detection confidence scores
    predict_score = []
    # store predicted detection results
    predict_anno = []
    # store predicted bounding-boxes
    predict_bbox = []

    HCDR_num = 0
    ####################################################################################################################
    fin_predict.readline()

    predict_num = int(fin_predict.readline().replace('\r','').replace('\n','').replace('\t',''))
    predict_index = predict_num

    ####################################################################################################################
    # predicted file of current image has no detection result
    if predict_num == 0:
        calibrated_num = 0
        fout_before.write(str(calibrated_num)+'\n')
        fout_after.write(str(calibrated_num)+'\n')
        for index in range(gt_num):
            fout.write("%d %d %d %d %d %d %d %d %d %d\n" % (gt_anno[index][0],gt_anno[index][1],gt_anno[index][2],gt_anno[index][3],gt_anno[index][4],gt_anno[index][5],gt_anno[index][6],gt_anno[index][7],gt_anno[index][8],gt_anno[index][9]))

        fin_predict.close()
        continue
    ####################################################################################################################

    while predict_index > 0:
        predict_anno_line = fin_predict.readline().split(" ")
        x = int(predict_anno_line[0])
        y = int(predict_anno_line[1])
        w = int(predict_anno_line[2])
        h = int(predict_anno_line[3])
        score = float(predict_anno_line[4])

        # only store high confidence detection results (HCDR)
        if score <= ADC:
            break

        HCDR_num = HCDR_num + 1

        # store the predicted detection confidence scores of these HCDRs
        predict_score.append(score)
        # store the predicted detection results of these HCDRs
        predict_anno.append([x,y,w,h])
        # store the predicted bounding-boxes of these HCDRs
        predict_bbox.append([x,y,x+w,y+h])

        predict_index = predict_index - 1

    ####################################################################################################################
    # Calculate the IoU betweeen predicted and annotated bounding-boxes
    # avoid predicted detection results has no HCDR, e.g. 0--Parade/0_Parade_Parade_0_939.jpg
    if HCDR_num != 0:
        # Calculate IoU
        overlaps=bbox_iou(np.array(predict_bbox),np.array(gt_bbox))

        argmax_overlaps = overlaps.argmax(axis=1)

        max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps]

        gt_argmax_overlaps = overlaps.argmax(axis=0)

        gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]

        ##############################################################################################
        # Determine calibration index

        # gt calibrated status, 0 uncalibrated, 1 calibrated
        # avoid multiple predicted bounding-boxes calibrating one misaligned annotation
        gt_calibrated = np.empty(len(gt_argmax_overlaps), dtype=np.int32)
        gt_calibrated.fill(0)
        # calibration index, -1 represents no need to calibrate (from predict perspective)
        calibrate = argmax_overlaps[:]

        for index in range(len(max_overlaps)):
            if max_overlaps[index] >= Tm:		  	    # predicted and annotated bounding-boxes matching
                if max_overlaps[index] > Tc:			# obtaining high localization accuracy, then no need to calibrate
                # high localization accuracy
                    calibrate[index] = -1
                else:
                # low localization accuracy, called misaligned detection result (MDR)
                    if gt_calibrated[argmax_overlaps[index]] == 0: 	# before uncalibrated, then calibrated (avoid multi predict for one gt)
                        gt_calibrated[argmax_overlaps[index]] = 1
                    else:						# before calibrated, then no need to calibrate
                        calibrate[index] = -1
            else:						# predicted and annotated bounding-boxes unmatching and no need to calibrate
                calibrate[index] = -1

        # the total num of bbox needed to be calibrated, facilitate to output calibrated bbox
        calibrated_num = 0

        for index in range(len(calibrate)):
            if calibrate[index]>=0:
                calibrated_num = calibrated_num + 1

        total_calibrated = total_calibrated + calibrated_num
        fout_before.write(str(calibrated_num)+'\n')
        fout_after.write(str(calibrated_num)+'\n')

        # For misaligned detection result MDR(Misaligned annotation, HCDR), calibrate misaligned annotations.
        # Replace operation: reasonably replace misaligned annotations with model predict bounding-boxs, marked as Replace(Misaligned bbx, High-confidence bbx)
        for index in range(len(calibrate)):
            if calibrate[index]>=0:
                # before: record misaligned annotations from original annotations
                fout_before.write("%d %d %d %d %d %d %d %d %d %d\n" % (gt_anno[calibrate[index]][0],gt_anno[calibrate[index]][1],gt_anno[calibrate[index]][2],gt_anno[calibrate[index]][3],gt_anno[calibrate[index]][4],gt_anno[calibrate[index]][5],gt_anno[calibrate[index]][6],gt_anno[calibrate[index]][7],gt_anno[calibrate[index]][8],gt_anno[calibrate[index]][9]))
                # predict form:(x,y,w,h,score,iou)
                # after: record replaced high confidence bounding-boxes from HCDRs
                fout_after.write("%d %d %d %d %f %f\n" % (predict_anno[index][0],predict_anno[index][1],predict_anno[index][2],predict_anno[index][3],predict_score[index],max_overlaps[index]))
                # replace operation
                gt_anno[calibrate[index]][0]=predict_anno[index][0]
                gt_anno[calibrate[index]][1]=predict_anno[index][1]
                gt_anno[calibrate[index]][2]=predict_anno[index][2]
                gt_anno[calibrate[index]][3]=predict_anno[index][3]
        # Store calibrated annotations
        for index in range(gt_num):
            fout.write("%d %d %d %d %d %d %d %d %d %d\n" % (gt_anno[index][0],gt_anno[index][1],gt_anno[index][2],gt_anno[index][3],gt_anno[index][4],gt_anno[index][5],gt_anno[index][6],gt_anno[index][7],gt_anno[index][8],gt_anno[index][9]))

        fin_predict.close()
##################################################################################################################################
    # if predicted detection results without any HCDR
    else:		#HCDR_num==0, no need to calibrate
        calibrated_num = 0
        fout_before.write(str(calibrated_num)+'\n')
        fout_after.write(str(calibrated_num)+'\n')
        for index in range(gt_num):
            fout.write("%d %d %d %d %d %d %d %d %d %d\n" % (gt_anno[index][0],gt_anno[index][1],gt_anno[index][2],gt_anno[index][3],gt_anno[index][4],gt_anno[index][5],gt_anno[index][6],gt_anno[index][7],gt_anno[index][8],gt_anno[index][9]))

        fin_predict.close()
    pass

# Output the number of being calibrated annotations
print("Total calibrated annotations: %d" % total_calibrated)

# Therefore, the calibrated annotations are stored in
# Calibrated annotations: wider_face_train_bbx_gt_new.txt

fout.close()
fout_before.close()
fout_after.close()
fin.close()
