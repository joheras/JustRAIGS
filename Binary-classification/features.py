import cv2
import argparse
import os
import sys
import pickle
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

def show(image, od, cup , trim, title):
 
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(title, fontsize=16)
    ax = axes.ravel()

    names = ['mask', 'disc', 'cup', 'trim']
    images = [image, od, cup, trim]
    for j, (im, names) in enumerate(zip(images, names)):
        ax[j].imshow(im, cmap='gray')
        ax[j].set_title(names)
        ax[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def sorted_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    scontours = sorted(contours, key=cv2.contourArea, reverse=True)
    return scontours

def filter(od, cup, trim):

    # Find contours in disc and sort by area
    # Check if the overalp area among cup and disc is area cup
    # Discard other segmented areas
    
    od_contours = sorted_contours(od)
    cup_contours = sorted_contours(cup)

    for od_contour in od_contours:
        mask_od = np.zeros(od.shape, dtype=np.uint8)
        cv2.drawContours(mask_od, [od_contour], 0, 1, -1)
        for cup_contour in cup_contours:
            mask_cup = np.zeros(od.shape, dtype=np.uint8)
            cv2.drawContours(mask_cup, [cup_contour], 0, 1, -1)

            if np.count_nonzero(mask_od*mask_cup) == np.count_nonzero(cup):
                od *= mask_od
                cup *= mask_od
                trim *= mask_od
                break

def globalFeatures(image, fDict):

    features = ['area', 'area_convex', 'area_filled', 'eccentricity', 'extent', 'orientation', 'solidity']

    label_image = label(image)
    for region in regionprops(label_image):
        for prop in features:
            fDict[prop] = region[prop]
        fDict['roundness'] = (region['perimeter']*region['perimeter'])/region['area']
        
        height = region.bbox[2] - region.bbox[0]
        width = region.bbox[3] - region.bbox[1]
        fDict['height'] = height
        fDict['width'] = width
        fDict['ratio_bbox'] = width/height

        major_axis = region.axis_major_length
        minor_axis = region.axis_minor_length
        fDict['major_axis'] = major_axis
        fDict['minor_axis'] = minor_axis
        try:
            fDict['ratio_bbox'] = major_axis/minor_axis
        except:
            print('error')

        

def ratioFeatures(fDict):
    cdr = ['area', 'area_convex', 'area_filled', 'major_axis', 'minor_axis', 'height', 'width']
    for measure in cdr:
        try:
        # if fDict['od']!= 0 and fDict['cup']!= 0:  # Verificar si el denominador no es cero
            fDict['cdr'][measure] = fDict['cup'][measure] / fDict['od'][measure]
        except:
            # Si el denominador es cero, manejarlo de alguna manera, por ejemplo, asignar un valor predeterminado o ignorarlo.
            # Aquí simplemente estamos imprimiendo un mensaje.
            print(f"La medida '{measure}' tiene un valor de área cero en 'od'. No se puede realizar la división.")
    #for measure in cdr:
     #   fDict['cdr'][measure] = fDict['cup'][measure]/fDict['od'][measure]

from tqdm import tqdm
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Features from cup and optic disc')

    parser.add_argument('--imageFolder', type=str, default='newMask2/')
    parser.add_argument('--gt', type=str, default='labels2.txt')
    parser.add_argument('--output', type=str, default='output/train_features_all.pkl')
    parser.add_argument('--output_csv', type=str, default='output/train_features_all.csv')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    df = pd.read_csv('prepared_binary_dataset.csv')
    df["pathMask"]=df["path"].apply(lambda x: x.replace("train/", "newMask2/").replace('.JPG','.png').replace('.PNG','.png'))
    # dfNew = pd.concat([df[(df.label=='training') & (df['Final Label']=='NRG')].sample(3000),
    #                df[(df.label=='training') & (df['Final Label']=='RG')],
    #                df[(df.label=='valid')]])
    features = dict()

    for name in tqdm(df.pathMask):
        # print(f"Processing {name}")
        filename = name
        base = os.path.splitext(os.path.basename(name))[0]
        # print(base)
        features[base] = dict()
        for key in ['od', 'cup', 'cdr']:
            features[base][key] = dict()
        # print(filename)
        image = cv2.imread(filename)
        
        od, cup, trim = cv2.split(image)
        # print(f"{image.shape} | od: {np.unique(od)} | cup: {np.unique(cup)} | trim: {np.unique(trim)}")

        # Discard incorrect segmented regions
        filter(od, cup, trim)
        if args.show:
            show(image, od, cup, trim, base)
        
        # Global features for od and cup
        globalFeatures(od, features[base]['od'])
        globalFeatures(cup, features[base]['cup'])
        # Rations between cup and od
        ratioFeatures(features[base])

        #print(features)


    # save dictionary to  pkl file
    with open(args.output, 'wb') as fp:
        pickle.dump(features, fp)
        print(f'Features saved successfully to {args.output}')


    # save dictionary to csv file
    # labels = pd.read_csv(args.gt)
    rows = []   
    for i, image in enumerate(features.keys()):

        # _, n = image.split("TRAIN")
        # df_label = labels.loc[labels['Image'] == int(n)]
        # label = 1 if 'G' in df_label['Label'].values else 0

        # if i == 0:
        #     fields = []
        #     for s in ['od', 'cup', 'cdr']:
        #         fields += [s + 'TRAIN' + e  for e in list(features[image][s].keys())]
        #     fields  += ['label']

        im_features = [image]+ list(features[image]['od'].values()) + list(features[image]['cup'].values()) + list(features[image]['cdr'].values()) 
        rows.append(im_features)


    with open(args.output_csv, 'w') as f:
        write = csv.writer(f)
        # write.writerow(fields)
        write.writerows(rows)
        print(f'Features saved successfully to {args.output_csv}')

        









       





