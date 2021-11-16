#Calculate the IoU between gt and predict
import numpy as np

def bbox_iou(gt, predict):
    N = gt.shape[0]
    K = predict.shape[0]
    overlaps = np.zeros((N, K))
    for k in range(K):
        box_area = ((predict[k,2]-predict[k,0]+1)*(predict[k,3]-predict[k,1]+1))
        for n in range(N):
            iw=(min(gt[n,2], predict[k,2])-max(gt[n,0], predict[k,0])+1)
            if iw>0:
                ih=(min(gt[n,3], predict[k,3])-max(gt[n,1], predict[k,1])+1)
                if ih>0:
                    ua=float((gt[n,2]-gt[n,0]+1)*(gt[n,3]-gt[n,1]+1)+box_area-iw*ih)
                    overlaps[n,k]=iw*ih/ua
    return overlaps
