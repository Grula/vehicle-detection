import sys
import pickle
import os 
import shutil

import lmdb
from path import Path

import cv2



commit_number = 1000

def store_many_lmdb(dataset, dataset_type):
    """ Stores an array of images to LMDB.
        Parameters:
        ---------------
        dataset      array of dicts : 
                        keys - image, label
    """
    # Load 100 images and take their avg 
    avg_mem = 0
    for d in dataset[:100]:
        avg_mem += sys.getsizeof(d['image']) + sys.getsizeof(d['label'])
    avg_mem = avg_mem // 100

    num_images = len(dataset)
    map_size = num_images * (1024 * (avg_mem))
    
    # Create a new LMDB DB for all the images
    shutil.rmtree(f"lmdb_{dataset_type}", ignore_errors= True)
    env = lmdb.open(str(f"lmdb_{dataset_type}"), map_size=map_size)
    txn = env.begin(write=True)

    # Same as before — but let's write all the images in a single transaction
    datas = []
    for i,d in enumerate(dataset):
        datas.append([f"{i:08}".encode('ascii'),
            {'image': d['image'] ,
             'label' : d['label']
            }
        ])
        if (i+1) % commit_number == 0:
            for d in datas:
                txn.put(d[0], pickle.dumps(d[1]))
            # print(f'Last unit stored {d[0]}')
            txn.commit()
            datas.clear()
            txn = env.begin(write=True)
    # Storing leftovers
    if datas:
        for d in datas:
            txn.put(d[0], pickle.dumps(d[1]))
        txn.commit()
        # print(f'Last unit stored {d[0]}')
        datas.clear()
    env.close()