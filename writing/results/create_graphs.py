# we want to load subfolders from current dir called eval 
# We read all text data files in folder (taking as much as one has)
# Calculate precision, recall, f1-score and save it to a table
# We save the table to a csv file

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
    files = os.listdir(folder)
    # get file that ends with csv
    files = [f for f in files if f.endswith('.csv')]
    if len(files) == 0:
        print("No .csv file found in this folder")
        continue
    
    for file in files:
        print("###################################################")
        print("Current file: ", file)
        file_path = os.path.join(folder, file)
        # Load data from file
        data = open(file_path, 'r')
        lines = data.readlines()[1:] # skip header
        lines = list(map(lambda x: x.strip().split(','), lines))
        data.close()



        mean_aps = {'car': [], 'motorbike': [], 'bus': [], 'truck': []}



        for class_name in classes:
            # filter lines by class name
            lines_class = list(filter(lambda x: class_name in x[1] , lines))
            # print("Number of lines for class: ", class_name, " is: ", len(lines_class))

            # for single class confusion martrix
            confusion_matrix = {'TP': 0, 'FP': 0,
                                'FN': 0, 'TN': 0}

            aps = {'precision': [], 'recall': [], 'f1': []}
            # we will get mAP for each class here
            
            for iou_threshold in np.arange(0.5, 1.0, 0.05):
                # print("Current IoU threshold: ", iou_threshold)
                for line in lines_class:
                    line = list(map(lambda x: x.strip(), line))
                    # get ground truth and prediction
                    gt = None if line[-2] == 'None' else line[-2]
                    pr = None if line[-1] == 'None' else line[-1]

                    if gt == None or pr == None:
                        iou_current = 0
                    else:
                        gt = list(map(int, line[-2].split(' ')))
                        pr = list(map(int, line[-1].split(' ')))
                        # calculate current IOU
                        iou_current = intersection_over_union(gt, pr)

                    # if IoU is greater than threshold we count it as true positive

                    # check if label is predicted correctly
                    true_label =  None if line[1] == 'None' else line[1]
                    pred_label =  None if line[2] == 'None' else line[2]
                    
                    # The model does not predict the label and is not a part of the ground truth.
                    if true_label == pred_label and pred_label is None:
                        confusion_matrix['TN'] += 1
                        continue
                    
                    if true_label == pred_label:
                        if iou_current > iou_threshold:# The model predicted a label and matches correctly as per ground truth.
                            confusion_matrix['TP'] += 1
                        else: # The model predicted a label, but it is not a part of the ground truth (Type I Error).
                            confusion_matrix['FP'] += 1
                    else: # The model does not predict a label, but it is part of the ground truth. (Type II Error).
                        confusion_matrix['FN'] += 1


                # calculate precision, recall and f1-score
                try:
                    precision = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FP'])
                except ZeroDivisionError:
                    precision = 0
                
                try:
                    recall = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FN'])
                except ZeroDivisionError:
                    recall = 0

                try:
                    f1 = 2 * (precision * recall) / (precision + recall)
                except ZeroDivisionError:
                    f1 = 0


                aps['precision'].append(precision)
                aps['recall'].append(recall)
                aps['f1'].append(f1)
            
            # calculate average f1-score for each class
            avg_f1 = sum(aps['f1']) / len(aps['f1'])
            print("Average f1-score for class: ", class_name, " is: ", avg_f1)
        

            # plot precision with iou threshold
            plt.plot(np.arange(0.5, 1.0, 0.05), aps['precision'], label=class_name)


        # plot precision with iou threshold
        plt.legend()
        plt.ylim(-0.05, 1.05)
        plt.xlabel('IoU prag')
        plt.ylabel('Preciznost')
        plt.title('Preciznost u odnosu na IoU prag za model ')
        plt.savefig(file.split('.')[0] + '.png')
        plt.close()

        # # calculate average precision for current file vs iou threshold
        # avg_precision = np.average(model_precision, axis=0)
        # # plot average precision for current file vs iou threshold
        # plt.plot(np.arange(0.5, 1.0, 0.05), avg_precision)
        # plt.ylim(-0.05, 1)
        # plt.xlabel('IoU prag')
        # plt.ylabel('Preciznost')
        # plt.title('Preciznost za ' + file_name.split('-')[0] + ' sa IoU pragom')
        # plt.savefig(file_name + '_avg.png')
        # plt.close()
