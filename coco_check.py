################################################################################
#@Author: Neel 
#Created on: 23-Nov-2022
# Example : 
#           
# python3 coco_check.py 

# Copyright (c) 2022 - Neelanjan Bhowmik, Durham University, UK
################################################################################
import numpy as np
import json
import sys
from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.patheffects as path_effects
# from matplotlib import rcParams
import argparse
import os
from tabulate import tabulate
################################################################################

def split_filename(filename):
    ext = ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', 
        '.tif', '.TIF', '.TIFF', '.tiff','.BMP', '.bmp',
        '.json', '.csv', '.txt', '.dat', '.vol']
    
    for suf in ext:
        if suf in filename:
            s = suf
            f, _ = filename.split(suf)
    return f, s
################################################################################

def json_stat(args):

    coco_ann_keys = [
        'id',
        'image_id',
        'category_id',
        'bbox',
        'area',
        'segmentation',
        'iscrowd'
    ]

    with open(args.jsonfile) as f:
        data = json.load(f)

    imgage_id = []
    cat_id = []
    n_x, n_y, n_w, n_h = 0, 0, 0, 0
    invalid_image = 0

    cat_stat = []
    area = []
    diff_keys_logs = []
    ####Category_id starts from '0' (coco format)
    catid_start = 1
    for cat in data['categories']:
        if cat['id'] == 0:
            catid_start = 0
        
        cat_id.append(cat['name'])
        cat_stat.append({
            "name": cat['name'],
            "id": cat['id'],
            "count": 0
            # "avg_width": 0,
            # "avg_height": 0,
            # "max_obj_area": [],
            # "max_width": [],
            # "max_height": [],
            # "min_obj_area": [],
            # "min_width": [],
            # "min_height": [],
        })

        area.append({
            "cat_id": cat['id'],
            "cat_name": cat['name'],
            "image_id": [],
            "area": [],
            "width": [],
            "height": [],
            "bbox_centre_x": [],
            "bbox_centre_y": []
        })
      
    diff_keys_logs.append(f'Missing keys in annotations id:')

    for i, ann in enumerate(tqdm(data['annotations'])):
        # print(ann.keys())
        ann_key = [keys for keys in ann]
        # print(ann_key)
        diff_keys = list(set(coco_ann_keys) - set(ann_key))
        if len(diff_keys) > 0:
            diff_keys_logs.append(f'{ann["id"]} {diff_keys}')
        # print(f'|__Missing keys in annotations id: {ann["id"]} {diff_keys}')

    
    bbox_log = ['Missing bbox in annotations id:']
    ann_log = ['Missing annotation in annotations id:']
    invalann_log = ['Invalid annotation in annotations id:']
    for i, ann in enumerate(tqdm(data['annotations'])):

        try:
            if len(ann['bbox']) < 4:
                bbox_log.append(f'{ann["id"]} {ann["bbox"]}')
                
            if ann['bbox'][2] > 0 and ann['bbox'][3] > 0:

                cat_stat[ann['category_id']-catid_start] = {
                    "name": cat_stat[ann['category_id']-catid_start]['name'],
                    "id": cat_stat[ann['category_id']-catid_start]['id'],
                    "count": cat_stat[ann['category_id']-catid_start]['count']+1,
                    # "avg_width": cat_stat[ann['category_id']-catid_start]['avg_width']+ann['bbox'][2],
                    # "avg_height": cat_stat[ann['category_id']-catid_start]['avg_height']+ann['bbox'][3],
                    # "max_obj_area": area[ann['category_id']-catid_start]['area'].append(ann['area']),
                    # "max_width": area[ann['category_id']-catid_start]['width'].append(ann['bbox'][2]),
                    # "max_height": area[ann['category_id']-catid_start]['height'].append(ann['bbox'][3]),
                    "image_id": area[ann['category_id']-catid_start]['image_id'].append(ann['image_id'])
                }   
            
            ####Stat target size
            imgage_id.append(ann['image_id'])
            ####Search for invalid annotation
            for im in data['images']:
                if im['id'] == ann['image_id']:
                    if ann['bbox'][0] < 0 or ann['bbox'][0] > im['width']:
                        n_x = 1+n_x
                        # invalid_image = 1+invalid_image
                        invalann_log.append(f'{ann["id"]} {im}')
                        
                    if ann['bbox'][1] < 0 or ann['bbox'][1] > im['height']:
                        n_y = 1+n_y
                        invalann_log.append(f'{ann["id"]} {im}')
                               
                    if ann['bbox'][2] <= 0 or ann['bbox'][2] > im['width']:
                        n_w = 1+n_w
                        invalann_log.append(f'{ann["id"]} {im}')
                        
                    if ann['bbox'][3] <= 0 or ann['bbox'][3] > im['height']:
                        n_h = 1+n_h
                        invalann_log.append(f'{ann["id"]} {im}')
                         
        
        except:
            ann_log.append(f'{ann["id"]} {sys.exc_info()}')
            # print(f'{ann}: {sys.exc_info()}')    
            pass    
  
    im_cnt = 0
    for i, cat in enumerate(cat_stat):
        # if cat['count'] != 0 : 
        #     avg_w = round(cat['avg_width']/cat['count'],2)
        #     avg_h = round(cat['avg_height']/cat['count'],2)
        # else:
        #     avg_w, avg_h = 0,0
  
        cat_stat[i] = {
            "name": cat['name'],
            "id": cat['id'],
            "count": cat['count'],
            "count%": round(cat["count"]/len(data["annotations"])*100,2)
            # "avg_width": avg_w,
            # "avg_height": avg_h,
            # "max_obj_area": max(area[i]["area"]) if len(area[i]["area"]) > 0 else 0,
            # "max_width": area[i]["width"][area[i]["area"].index(max(area[i]["area"]))] if len(area[i]["area"]) > 0 else 0,
            # "max_height": area[i]["height"][area[i]["area"].index(max(area[i]["area"]))] if len(area[i]["area"]) > 0 else 0,
            # "min_obj_area": min(area[i]["area"]) if len(area[i]["area"]) > 0 else 0,
            # "min_width": area[i]["width"][area[i]["area"].index(min(area[i]["area"]))] if len(area[i]["area"]) > 0 else 0,
            # "min_height": area[i]["height"][area[i]["area"].index(min(area[i]["area"]))] if len(area[i]["area"]) > 0 else 0
        }

    sum_log = []
    sum_log.append('Annotation statistics:')
    sum_log.append(f'Total Images: {len(data["images"])}')
    sum_log.append(f'Total Instances: {len(data["annotations"])}')
    sum_log.append(f'Total Class: {len(cat_stat)}')
    
    print('\nAnnotation statistics:')
    
    print(json.dumps(cat_stat, indent=4, sort_keys=True))
    
    # print(f'\nInvald x:{n_x}\nInvald y:{n_y}\nInvald w:{n_w}\nInvald h:{n_h}')
    # tmp_txt = f'Invald x:{n_x}\nInvald y:{n_y}\nInvald w:{n_w}\nInvald h:{n_h}'
    # invalann_log.append(tmp_txt)
    
    print('\nTotal Images: ',len(data['images']))
    print('Total Instances: ',len(data['annotations']))
    print('Total Class: ',len(cat_stat))

    pf = True
    if len(diff_keys_logs) <=1:
        if len(bbox_log) <=1:
            if len(ann_log) <=1:
                if len(invalann_log) <=1:
                    pf = True

    else:
        pf = False
    
    print('\n')
    if pf == True:
        print('Annotation check: [Pass]')
        
    else:
        print('Annotation check: [Fail]')
        print('Refer logs for more details')

    with open(args.logfile, 'w') as f:
        for val in diff_keys_logs: 
            f.write(val)
            f.write('\n')
        
        f.write('\n\n')
        
        for val in bbox_log: 
            f.write(val)
            f.write('\n')

        f.write('\n\n')

        for val in ann_log: 
            f.write(val)
            f.write('\n')
        
        f.write('\n\n')

        for val in invalann_log: 
            f.write(val)
            f.write('\n')
        
        f.write('\n\n')

        for val in sum_log: 
            f.write(val)
            f.write('\n')
        f.write('\n\n')
        if pf == True:
            f.write('Annotation check: [Pass]')
        if pf == False:
            f.write('Annotation check: [Fail]')

    f.close()
    print('\n[Done]')

    return cat_stat

################################################################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonfile", 
        type=str, 
        help="Input json file path")
    parser.add_argument("--logfile", 
        type=str, 
        default='logs/logs.out',
        help="Log file path")
    args = parser.parse_args()
    return args

################################################################################

def main():
    args = parse_args()
    t_val = []
    for arg in vars(args):
        t_val.append([arg, getattr(args, arg)])
    print('\n')
    print(tabulate(t_val,
                ['input', 'value'],
                tablefmt="psql"))


    logd, logf = os.path.split(args.logfile)
    os.makedirs(logd, exist_ok=True)

    if args.jsonfile:
        cat_stat = json_stat(args)
    else:
        print(f'Mising annotation file: {args.jsonfile}')
    
################################################################################

if __name__ == '__main__':
    main()

####