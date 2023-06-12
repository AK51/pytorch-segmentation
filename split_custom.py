import os
import random
import re
from PIL import Image, ImageFile
from tqdm import tqdm
import numpy as np
import argparse

ImageFile.LOAD_TRUNCATED_IMAGES = True

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True,
                help="path to images")

ap.add_argument("-m", "--masks", type=str, required=True,
                help="path to your masks")

ap.add_argument("-o", "--output", type=str, required=True,
                help="path to where the split dataset should be stored")

ap.add_argument("--image-format", dest="image_format", type=str, default="jpg",
                help="image format, defaults to jpg")

ap.add_argument("--mask-format", dest="mask_format", type=str, default="png",
                help="mask format, defaults to png")

ap.add_argument("--keep-original", dest="keep_original", action="store_true",
                help="keep the original images after storing them into corresponding folders")


args = vars(ap.parse_args())


# Variables (change if needed)

INPUT_IMAGE_PATH = args["images"]
INPUT_MASK_PATH = args["masks"]

OUTPUT_DATA_PATH = args["output"]
OUTPUT_IMAGE_PATH = OUTPUT_DATA_PATH+'/images'
OUTPUT_MASK_PATH = OUTPUT_DATA_PATH+'/annotations'
IMAGE_FORMAT = '.' + args["image_format"]
MASK_FORMAT = '.' + args["mask_format"]

# Remove duplicates after adding them to train/val folders
keep_old_images = args["keep_original"]

# Get all images and masks, sort them and shuffle them to generate data sets.

all_masks = [os.path.splitext(x)[0] for x in os.listdir(
    INPUT_MASK_PATH) if MASK_FORMAT in os.path.splitext(x)[1]]


all_images = [os.path.splitext(x)[0] for x in os.listdir(
    INPUT_IMAGE_PATH) if IMAGE_FORMAT in os.path.splitext(x)[1] and os.path.splitext(x)[0] in all_masks]


all_images.sort(key=lambda var: [int(x) if x.isdigit() else x
                                 for x in re.findall(r'[^0-9]|[0-9]+', var)])
all_masks.sort(key=lambda var: [int(x) if x.isdigit() else x
                                for x in re.findall(r'[^0-9]|[0-9]+', var)])


random.seed(230)
random.shuffle(all_images)


# Split images to train,val,test sets (80% : 10%  : 10% ratio)

split = int(0.1*len(all_images))

b_images = [None] * 10
b_images[0] = all_images[:split]
b_images[1] = all_images[split:split*2]
b_images[2] = all_images[split*2:split*3]
b_images[3] = all_images[split*3:split*4]
b_images[4] = all_images[split*4:split*5]
b_images[5] = all_images[split*5:split*6]
b_images[6] = all_images[split*6:split*7]
b_images[7] = all_images[split*7:split*8]
b_images[8] = all_images[split*8:split*9]
b_images[9] = all_images[split*9:]

for i in range(10):
    val_images = b_images[i]
    test_images = b_images[(i+1)%10]
    train_images = b_images[(i+2)%10] + b_images[(i+3)%10] + b_images[(i+4)%10] + b_images[(i+5)%10] + b_images[(i+6)%10] + b_images[(i+7)%10] + b_images[(i+8)%10] + b_images[(i+9)%10]
    OUTPUT_DATA_PATH = args["output"] + str(i)
    OUTPUT_IMAGE_PATH = OUTPUT_DATA_PATH+'/images'
    OUTPUT_MASK_PATH = OUTPUT_DATA_PATH+'/annotations'

    print(
        f'SPLIT: {len(train_images)} train , {len(val_images)} valuation and {len(test_images)} testing images!')
    print('-------------------------------------------------------------------------------')


    # Generate corresponding mask lists for masks


    train_masks = [f for f in all_masks if f in train_images]
    val_masks = [f for f in all_masks if f in val_images]
    test_masks = [f for f in all_masks if f in test_images]


    # Generate required folders

    train_folder = 'training'
    val_folder = 'validation'
    test_folder = 'testing'

    folders = [train_folder, val_folder, test_folder]

    for folder in folders:
        os.makedirs(os.path.join(OUTPUT_IMAGE_PATH, folder), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_MASK_PATH, folder), exist_ok=True)


    # Add train, val and test images and their masks to corresponding folders


    def add_images(dir_name, image):

        full_image_path = INPUT_IMAGE_PATH+'/'+image+IMAGE_FORMAT
        img = Image.open(full_image_path)
        img = img.convert("RGB")
        img.save(OUTPUT_IMAGE_PATH+'/{}'.format(dir_name)+'/'+image+IMAGE_FORMAT)

        if not keep_old_images:
            os.remove(full_image_path)


    def add_masks(dir_name, image):

        full_mask_path = INPUT_MASK_PATH+'/'+image+MASK_FORMAT
        img = Image.open(full_mask_path)

        img.save(OUTPUT_MASK_PATH+'/{}'.format(dir_name)+'/'+image+MASK_FORMAT)

        if not keep_old_images:
            os.remove(full_mask_path)


    image_folders = [(train_images, train_folder), (val_images, val_folder), (test_images, test_folder)]

    mask_folders = [(train_masks, train_folder), (val_masks, val_folder), (test_masks, test_folder)]

    print(
        f'Writing images to the {image_folders[0][1]} and {image_folders[1][1]} folders...')

    for folder in image_folders:

        array = folder[0]
        name = [folder[1]] * len(array)

        list(map(add_images, tqdm(name), array))

    print(
        f'Writing masks to the {mask_folders[0][1]} and {mask_folders[1][1]} folders...')

    for folder in mask_folders:

        array = folder[0]
        name = [folder[1]] * len(array)

        list(map(add_masks, tqdm(name), array))

print('Done!')
