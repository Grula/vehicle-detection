"""Demo for use yolo v3
"""
import os
import time
import cv2
import numpy as np
from yolo_model import YOLO


def process_image(img):
    """Resize, reduce and expand image.

    # Argument:
        img: original image.

    # Returns
        image: ndarray(64, 64, 3), processed image.
    """
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image


def get_classes(file):
    """Get classes name.

    # Argument:
        file: classes name for database.

    # Returns
        class_names: List, classes name.

    """
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names


def draw(image, boxes, scores, classes, all_classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 1,
                    cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

    print()


def detect_image(image, yolo, all_classes):
    """Use yolo v3 to detect images.

    # Argument:
        image: original image.
        yolo: YOLO, yolo model.
        all_classes: all classes name.

    # Returns:
        image: processed image.
    """
    pimage = process_image(image)

    start = time.time()
    boxes, classes, scores = yolo.predict(pimage, image.shape)
    end = time.time()

    print('time: {0:.2f}s'.format(end - start))

    if boxes is not None:
        draw(image, boxes, scores, classes, all_classes)

    return image , boxes, classes, scores


if __name__ == '__main__':
    yolo = YOLO(0.8, 0.7)
    file = 'coco_classes.txt'
    all_classes = get_classes(file)

    images_path = 'augmented_gan100k/'
    # detect images in test floder
    # create csv file
    f = open('data_augmented.csv', 'w')
    # im_path, label, x, y, w, h
    f.write('im_path,label,x,y,w,h\n')
    subfolders = [ f.path for f in os.scandir(images_path) if f.is_dir() ]
    for subfolder in subfolders:
        images = [f for f in os.listdir(subfolder) if f.endswith('.png')]
        current_label = subfolder.split('/')[-1]
        destination_folder = f'{subfolder}-PREDICTED'
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        for image_file in images:
            image_path = os.path.join(subfolder, image_file)
            image = cv2.imread(image_path)

            print("Detecing image: {}".format(image_path))
            image, boxes, scores, classes = detect_image(image, yolo, all_classes)

            #find image with highest score if exists
            if scores is not None and len(scores) > 0:
                max_score = 0
                max_idx = 0
                for i, box in enumerate(boxes):
                    if scores[i] > max_score:
                        max_score = scores[i]
                        max_idx = i

                x0, y0, x1, y1 = boxes[max_idx]
                x = max(0, np.floor(x0 + 0.5).astype(int))
                y = max(0, np.floor(y0 + 0.5).astype(int))
                right = min(image.shape[1], np.floor(x1 + 0.5).astype(int))
                bottom = min(image.shape[0], np.floor(y1 + 0.5).astype(int))
                w = right - x
                h = bottom - y


                # saving info to csv file 
                f.write(f'{image_path},{current_label},{x},{y},{w},{h}\n')

                cv2.imwrite(f'{destination_folder}/{image_file}', image)
                print(f'Saved {image_file}')    

    # yolo.train()
    # images_path = 'data/valid'
    # images_path = 'images'
    # # detect images in test floder.
    # for (root, dirs, files) in os.walk(images_path):
    #     if files:
    #         for f in files:
    #             print(f)
    #             path = os.path.join(root, f)
    #             image = cv2.imread(path)
    #             image = detect_image(image, yolo, all_classes)
    #             cv2.imwrite('pred_images/' + f, image)

    
