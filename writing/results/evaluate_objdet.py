# we want to load subfolders from current dir called eval 
# We read all text data files in folder (taking as much as one has)
# Calculate precision, recall, f1-score and save it to a table
# We save the table to a csv file

import os
import numpy as np

#import plt
from matplotlib import pyplot as plt

SMOOTH = 1e-6

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

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


        confusion_matrix = {'car': {'car': 0, 'motorbike': 0, 'bus': 0, 'truck': 0,},
                            'motorbike': {'car': 0, 'motorbike': 0, 'bus': 0, 'truck': 0,},
                            'bus': {'car': 0, 'motorbike': 0, 'bus': 0, 'truck': 0,},
                            'truck': {'car': 0, 'motorbike': 0, 'bus': 0, 'truck': 0,},
                            }

        confusion_matrix_TF = { 'car': {'TP': 0, 'FN': 0, 'FP': 0, 'FN': 0},
                                'motorbike': {'TP': 0, 'FN': 0, 'FP': 0, 'FN': 0},
                                'bus': {'TP': 0, 'FN': 0, 'FP': 0, 'FN': 0},
                                'truck': {'TP': 0, 'FN': 0, 'FP': 0, 'FN': 0},
                                }
        # We will store IoU results in this array for each preidction
        IoU_class_res = {'car': [], 'motorbike': [], 'bus': [], 'truck': []}

        # Calculate confusion matrix for each class
        for line in lines:
            line = list(map(lambda x: x.strip(), line)) # get rid of whitespaces
            image_path, true_class, predicted_class, score, true_bbox, predicted_bbox = line

            if predicted_class == 'None':
                confusion_matrix_TF[true_class]['FN'] += 1
            else:
                confusion_matrix[true_class][predicted_class] += 1

            # Calculate IoU for each prediction
            # convert boxes fro x,y,w,h to x1,y1,x2,y2
            if predicted_bbox != 'None':
                true_bbox = list(map(int, true_bbox.split(' ')))
                predicted_bbox = list(map(int, predicted_bbox.split(' ')))
                true_bbox = [true_bbox[0], true_bbox[1], true_bbox[0] + true_bbox[2], true_bbox[1] + true_bbox[3]]
                predicted_bbox = [predicted_bbox[0], predicted_bbox[1], predicted_bbox[0] + predicted_bbox[2], predicted_bbox[1] + predicted_bbox[3]]
                IoU_class_res[true_class].append(bb_intersection_over_union(true_bbox, predicted_bbox))
            else:
                IoU_class_res[true_class].append(0)


        # pirnt Confusion matrix
        print("Confusion matrix:")
        print("\t", end='')
        for class_name in classes:
            print(class_name, end='\t')
        print("")
        for class_name in classes:
            print(class_name, end='\t')
            for class_name2 in classes:
                print(confusion_matrix[class_name][class_name2], end='\t')
            print("")
        print("")
        


        # For each class calculate TP, FP, TN, FN
        for i, (true_class, predictions) in enumerate(confusion_matrix.items()):
            # Calculate TP
            confusion_matrix_TF[true_class]['TP'] = predictions[true_class]
            # Calculate FN
            confusion_matrix_TF[true_class]['FN'] = sum(predictions.values()) - predictions[true_class]
            # Calculate FP
            for predicted_class in classes:
                if predicted_class != true_class:
                    confusion_matrix_TF[true_class]['FP'] += confusion_matrix[predicted_class][true_class]
            # Calculate TN
            confusion_matrix_TF[true_class]['TN'] = sum([sum(row.values()) for row in confusion_matrix.values()]) - \
                                                    confusion_matrix_TF[true_class]['TP'] - \
                                                    confusion_matrix_TF[true_class]['FP'] - \
                                                    confusion_matrix_TF[true_class]['FN']

        # pirnt Confusion matrix
        # print("Postivies matrix for ", file)
        # print("\t", end='')
        # for class_name in classes:
        #     print(class_name, end='\t')
        # print()
        # for class_name, row in confusion_matrix_TF.items():
        #     print(class_name, end='\t')
        #     for value in row.values():
        #         print(value, end='\t')
        #     print()
        # print()
              

        # precision is calculated based on the IoU threshold. for BBOXES 
        iou_ap = []
        for iou_threshold in np.arange(0.5, 1.0, 0.01):
            # Calculate precision for each class
            precision = {'car': 0, 'motorbike': 0, 'bus': 0, 'truck': 0}
            for true_class, IoU_results in IoU_class_res.items():
                precision[true_class] = sum([1 for IoU in IoU_results if IoU >= iou_threshold]) / len(IoU_results)
            precision_all = sum(precision.values()) / len(classes)
            iou_ap.append(precision_all)
            # print("Precision for IoU threshold: ", iou_threshold, " is: ", precision_all)

        # Calculate mAP
        mAP = sum(iou_ap) / len(iou_ap)
        print("mAP: ", mAP)

        # plit iou_curve
        iou_ap = np.array(iou_ap)
        plt.plot(np.arange(0.5, 1.0, 0.01), iou_ap)
        plt.xlabel('IoU Threshold')
        plt.ylabel('Precision')
        plt.title(f'Precision vs IoU Threshold for {file.split("-")[0]}')
        # plt.show()
        plt.savefig(os.path.join(folder, f'precision_vs_iou_threshold_{file.split("-")[0]}.png'))
        plt.close()
        # Precision score = sum(TP) / (sum(TP) + sum(FP))
        precision = {'car': 0, 'motorbike': 0, 'bus': 0, 'truck': 0}
        for true_class in classes:
            try:
                precision[true_class] = confusion_matrix_TF[true_class]['TP'] / (confusion_matrix_TF[true_class]['TP'] + confusion_matrix_TF[true_class]['FP'])
            except ZeroDivisionError:
                precision[true_class] = 0
        precision_all = sum(precision.values()) / len(classes)

        # Recall score = sum(TP) / (sum(TP) + sum(FN)) 
        recall = {'car': 0, 'motorbike': 0, 'bus': 0, 'truck': 0}
        for true_class in classes:
            try:
                recall[true_class] = confusion_matrix_TF[true_class]['TP'] / (confusion_matrix_TF[true_class]['TP'] + confusion_matrix_TF[true_class]['FN'])
            except ZeroDivisionError:
                recall[true_class] = 0
        recall_all = sum(recall.values()) / len(classes)

        # calculate F1 score for each class
        F1 = {'car': 0, 'motorbike': 0, 'bus': 0, 'truck': 0}
        for true_class in classes:
            try:
                F1[true_class] = 2 * precision[true_class] * recall[true_class] / (precision[true_class] + recall[true_class])
            except ZeroDivisionError:
                F1[true_class] = 0

        F1_all = 2 * precision_all * recall_all / (precision_all + recall_all)
        print("Precision for classes: ", precision_all)
        print("Recall for classes: ", recall_all)
        print("F1 for classes:: ", F1_all)

        # plot precision recall curve





    # The area is the sum of all individual precisions multiplied by the individual recalls.
    # The individual precisions are the TP / (TP + FP)
    # The individual recalls are the TP / (TP + FN)
    



