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

        precision = metrics.precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive", zero_division=0)
        recall = metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label="positive", zero_division=0)
        
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
fw_mAP = open('mAPs.txt', 'w')
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

        mean_aps = {'car': 0, 'motorbike': 0, 'bus': 0, 'truck': 0}

        # Write name of the file to the file with mAPs
        fw_mAP.write(f"------- {file.split('.')[0]} -------\n")

        for class_name in classes:
            # filter lines by class name
            lines_class = list(filter(lambda x: class_name in x[1] , lines))
            # print("Number of lines for class: ", class_name, " is: ", len(lines_class))            

            y_true = []
            pred_scores = []

            for line in lines_class:
                line = list(map(lambda x: x.strip(), line))

                # check if label is predicted correctly
                true_label =  None if line[1] == 'None' else line[1]
                pred_label =  None if line[2] == 'None' else line[2]

                if pred_label == None: # that means there are  no bboxes for this image
                    iou_current = 0
                else:
                    gt = line[-2]
                    pr = line[-1]

                    gt = list(map(int, line[-2].split(' ')))
                    pr = list(map(int, line[-1].split(' ')))
                    # calculate current IOU
                    iou_current = intersection_over_union(gt, pr)


                # if label is predicted correctly
                if true_label == pred_label:
                    y_true.append('positive')
                else:
                    y_true.append('negative')

                pred_scores.append(iou_current)
            
             
            # Calculate AP for current class
            thresholds = np.arange(start=0.5, stop=1, step=0.05)
            precisions, recalls = precision_recall_curve(y_true=y_true, 
                                             pred_scores=pred_scores, 
                                             thresholds=thresholds)
            precisions.append(1.0)
            recalls.append(0.0)
            # convert to numpy array
            precisions = np.array(precisions, dtype=np.float32)
            recalls = np.array(recalls, dtype=np.float32)

            AP = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
            print(f"AP for {class_name}: {AP}")
            mean_aps[class_name] = AP
            fw_mAP.write(f"AP for {class_name}: {AP}\n")


        #     # Plot precision-recall curve
        #     plt.plot(recalls, precisions, label=class_name)
        
        # # plot all classes together
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.title('Precision-Recall curve for {}'.format(file.split('.')[0]))
        # plt.legend(loc="lower left")
        # plt.show()




        # Calculate mAP for current file
        mAP = np.mean(list(mean_aps.values()))
        print("mAP: ", mAP)
        fw_mAP.write(f"mAP for overall: {mAP}\n")

fw_mAP.close()