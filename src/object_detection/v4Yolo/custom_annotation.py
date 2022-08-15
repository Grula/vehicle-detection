import csv


# open csv and save as txt file
# csv_paths = ['data/data.csv', 'data/augmented_stylegan_bt.csv', 'data/augmented_stylegan_cm.csv' ,]
csv_paths = ['data/data.csv', 'data/augmented_basic.csv',]

train_annotation_path = 'data/train_data.txt'
valid_annotation_path = 'data/valid_data.txt'

custom_id = {'car' : 0, 'motorbike' : 1, 'bus' : 2, 'truck' : 3}

ftrain = open(train_annotation_path, 'w')
fvalid = open(valid_annotation_path, 'w')
for csv_path in csv_paths:
    print(csv_path)
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        for row in csv_reader:
            filename, label, startX, startY, w, h = row
            # filename bbox, label_id
            endX = int(startX) + int(w)
            endY = int(startY) + int(h)
            if 'valid' in filename:
                fvalid.write('{} {},{},{},{},{}\n'.format(filename, startX, startY, endX, endY, custom_id[label]))
            else:
                ftrain.write('{} {},{},{},{},{}\n'.format(filename, startX, startY, endX, endY, custom_id[label]))

ftrain.close()
fvalid.close()