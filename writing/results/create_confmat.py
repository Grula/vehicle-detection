
from ast import Gt
import os
import numpy as np
from sklearn import metrics

#import plt
from matplotlib import pyplot as plt

SMOOTH = 1e-6

def precision_recall_curve(y_true, pred_scores, thresholds):
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        y_pred = ["positive" if score >= threshold else "negative" for score in pred_scores]

        precision = metrics.precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
        recall = metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
        
        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls

def intersection_over_union(gt_box, pred_box):
    inter_box_top_left = [max(gt_box[0], pred_box[0]), max(gt_box[1], pred_box[1])]
    inter_box_bottom_right = [min(gt_box[0]+gt_box[2], pred_box[0]+pred_box[2]), min(gt_box[1]+gt_box[3], pred_box[1]+pred_box[3])]

    inter_box_w = inter_box_bottom_right[0] - inter_box_top_left[0]
    inter_box_h = inter_box_bottom_right[1] - inter_box_top_left[1]

    intersection = inter_box_w * inter_box_h
    union = gt_box[2] * gt_box[3] + pred_box[2] * pred_box[3] - intersection
    
    iou = intersection / union

    return iou
    return iou, intersection, union





# Object detection folders
folders = ['vgg16', 'yolo' ]

# In each folder there is file that ends with .csv

classes = ['car', 'motorbike', 'bus', 'truck']
for folder in folders:
    print("Current Object Detection folder: ", folder)
    csv_files = os.listdir(folder)
    # get file that ends with csv
    csv_files = [f for f in csv_files if f.endswith('.csv')]
    if len(csv_files) == 0:
        print("No .csv file found in this folder")
        continue
    for csvf in csv_files:

        if 'NO' not in csvf:
            continue
        
        print("###################################################")
        print("Current file: ", csvf)
        file_path = os.path.join(folder, csvf)
        # Load data from file
        data = open(file_path, 'r')
        lines = data.readlines()[1:] # skip header
        lines = list(map(lambda x: x.strip().split(','), lines))
        data.close()

        print(len(lines))

        _dict_data = {}
        for line in lines:
            if line[0] not in _dict_data:
                _dict_data[line[0]] = []
            _dict_data[line[0]].append(line[1:])

  
  
        score_threshold = 0.5

        filter_score_data = {}
        for img_path, img_data in _dict_data.items():
            # filter low scores
            for _d in img_data:
                score = float(_d[2])
                if score >= score_threshold:
                    if img_path not in filter_score_data:
                        filter_score_data[img_path] = []
                    filter_score_data[img_path].append(_d)
        print("Number of images: ", len(filter_score_data))

        confusion_matrix = {'car': {'car': 0, 'motorbike': 0, 'bus': 0, 'truck': 0},
                                    'motorbike': {'car': 0, 'motorbike': 0, 'bus': 0, 'truck': 0},
                                    'bus': {'car': 0, 'motorbike': 0, 'bus': 0, 'truck': 0}, 
                                    'truck': {'car': 0, 'motorbike': 0, 'bus': 0, 'truck': 0}
                                    }

        confusion_matrix = np.zeros( (len(classes)+1, len(classes))+1 )
        filter_iou_data = {}
        iou_threshold = 0.5
        for img_path, img_data in filter_score_data.items():
            # calculate iou
            for _d in img_data:
                gt = _d[-2]
                pred = _d[-1]

                gt_box = [int(float(x)) for x in gt.strip().split(' ')]
                pred_box = [int(float(x)) for x in pred.strip().split(' ')]
                iou_current = intersection_over_union(gt_box, pred_box)
                if iou_current >= iou_threshold:
                    if img_path not in filter_iou_data:
                        filter_iou_data[img_path] = {'true_label': _d[0], 'pred_label': _d[1], 'iou_score':iou_current, 'gt': gt, 'pred': pred}
                    else:
                        if filter_iou_data[img_path]['iou_score'] < iou_current:
                            filter_iou_data[img_path] = {'true_label': _d[0], 'pred_label': _d[1], 'iou_score':iou_current, 'gt': gt, 'pred': pred}
        print("Number of images after iou filter: ", len(filter_iou_data))

        # for line in lines:
            # line = list(map(lambda x: x.strip(), line))
            # # get ground truth and prediction
            # gt = None if line[-2] == 'None' else line[-2]
            # pr = None if line[-1] == 'None' else line[-1]

            # if gt == None or pr == None:
            #     iou_current = 0
            # else:
            #     gt = list(map(int, line[-2].split(' ')))
            #     pr = list(map(int, line[-1].split(' ')))
            #     # calculate current IOU
            #     iou_current = intersection_over_union(gt, pr)

            
            # # if IoU is greater than threshold we count it as true positive

            # # check if label is predicted correctly
            # true_label =  None if line[1] == 'None' else line[1]
            # pred_label =  None if line[2] == 'None' else line[2]
            
            
            # # The model does not predict the label and is not a part of the ground truth.
            # if true_label is None or pred_label is None:
            #     continue

            # if true_label == pred_label:
            #     if iou_current > iou_threshold:
            #         confusion_matrix[true_label][pred_label] += 1
            # else:
            #     confusion_matrix[true_label][pred_label] += 1



        # print confusion matrix nicely
        print("Confusion Matrix:")
        print("{:<10} {:<10} {:<10} {:<10} {:<10}".format("", "car", "motorbike", "bus", "truck"))
        for cls in classes:
            print("{:<10} {:<10} {:<10} {:<10} {:<10}".format(cls, confusion_matrix[cls]['car'], confusion_matrix[cls]['motorbike'], confusion_matrix[cls]['bus'], confusion_matrix[cls]['truck']))
        print("###################################################")
        print("")
        exit()
