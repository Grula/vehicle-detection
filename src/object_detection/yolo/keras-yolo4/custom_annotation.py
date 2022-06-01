import csv


# open csv and save as txt file
csv_path = './data/data.csv'
train_annotation_path = './data/train_data.txt'
valid_annotation_path = './data/valid_data.txt'
custom_id = {'car' : 0, 'motorbike' : 1, 'bus' : 2, 'truck' : 3}
ft = open(train_annotation_path, 'w')
fv = open(valid_annotation_path, 'w')
with open(csv_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader, None)
    for row in csv_reader:
        filename, label, startX, startY, w, h = row
        # filename bbox, label_id
        endX = int(startX) + int(w)
        endY = int(startY) + int(h)
        if 'valid' in filename:
            fv.write('{} {},{},{},{},{}\n'.format(filename, startX, startY, endX, endY, custom_id[label]))
        else:
            ft.write('{} {},{},{},{},{}\n'.format(filename, startX, startY, endX, endY, custom_id[label]))

ft.close()
fv.close()