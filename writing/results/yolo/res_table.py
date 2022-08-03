# we want to load subfolders from current dir called eval 
# We read all text data files in folder (taking as much as one has)
# Calculate precision, recall, f1-score and save it to a table
# We save the table to a csv file


import os



# Traverse current folders and subfolders in search of folder eval
folders = [ f.path for f in os.scandir('.') if f.is_dir() ]

res_data = {}
# custom_id = {'car' : 0, 'motorbike' : 1, 'bus' : 2, 'truck' : 3}
custom_id = {'0' : 'car', '1' : 'motorbike', '2' : 'bus', '3' : 'truck'}

min_files = None
# search folder in folders for dir called eval
for folder in folders:
    subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
    # check if list has dir that contains name 'eval'
    for subf in subfolders:
        if 'eval' not in subf:
            continue
        exp_name = subf.split('/')[1]
        # get all files in subfolder
        files = [f for f in os.listdir(subf) if f.endswith('.txt')]
        files.sort(key = lambda s: int(s.split('_')[2].split('.')[0]))
        # print(files)
        res_data[exp_name] = {'files' :files}
        if min_files is None:
            min_files = len(files)
        elif min_files > len(files):
            min_files = len(files)


# Chop the extra files
for exp_name in res_data:
    files = res_data[exp_name]['files']
    files = files[:min_files]
    res_data[exp_name]['files'] = files


# Load all files and calculate precision, recall, f1-score
for exp_name in res_data:
    for file in res_data[exp_name]['files']:
        file_path = os.path.join(exp_name, 'eval' ,file)

        # load file
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # line consist of, path_to_img bboxes, probability, class_id
            for line in lines:
                
                line = line.strip().split('  ')
                path = line[0]
                bboxes = line[1:][0]

                coords = bboxes.split(',')
                
                class_id = coords[-1]
                prob = coords[-2]

                if class_id not in res_data[exp_name]:
                    res_data[exp_name][class_id] = [prob,]
                else:
                    res_data[exp_name][class_id].append(prob)


                # bboxes can containt [coorinates],probability, class_id
                break
        break


# from pprint import pprint
# pprint(res_data)