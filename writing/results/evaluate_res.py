# we want to load subfolders from current dir called eval 
# We read all text data files in folder (taking as much as one has)
# Calculate precision, recall, f1-score and save it to a table
# We save the table to a csv file


import os
import numpy as np


# Traverse current folders and subfolders in search of folder eval
folders = [ f.path for f in os.scandir('.') if f.is_dir() ]

res_data = {}
# custom_id = {'car' : 0, 'motorbike' : 1, 'bus' : 2, 'truck' : 3}
custom_id = {'0' : 'car', '1' : 'motorbike', '2' : 'bus', '3' : 'truck'}

min_files = None
# search folder in folders for dir called eval
types = {}
for folder in folders:
    files = os.listdir(folder)
    files = [os.path.join(folder, f) for f in files]
    types[folder.split('-')[-1]] = list(filter(lambda x: x.endswith('.csv'), files))


    


for key,path in types.items():
    #load csv file
    data = {}
    with open(path[0], 'r') as f:
        f.readline() # skip header

        for line in f.readlines():
            line = line.strip().split(',')
            label, correctly_pred = line[0].split(':')
            score = float(line[1])
            if label not in data:
                data[label] = np.array([True if correctly_pred == 'True' else False])
            else:
                data[label] = np.append( data[label], bool(True if correctly_pred == 'True' else False))
    

    # # precision
    print(key.split('-')[-1])
    print('car: ',data['car'].sum(),'/', len(data['car']))
    print('motorbike: ',data['motorbike'].sum(),'/', len(data['motorbike']))
    print('bus: ',data['bus'].sum(),'/', len(data['bus']))
    print('truck: ',data['truck'].sum(),'/', len(data['truck']))




    # prec = data['car'].sum() / float(len(data['car']))
    # prec = data['motorbike'].sum() / float(len(data['motorbike']))
    # print('motorbike:',prec)
    # prec = data['bus'].sum() / float(len(data['bus']))
    # print('bus:',prec)
    # prec = data['truck'].sum() / float(len(data['truck']))
    # print('truck:',prec)



