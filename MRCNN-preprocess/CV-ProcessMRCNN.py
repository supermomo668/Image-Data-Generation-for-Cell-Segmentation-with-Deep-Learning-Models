#!/usr/bin/env python
# coding: utf-8

# In[48]:

import os,  numpy as np, random, skimage.io, matplotlib.pyplot as plt, skimage.exposure
import numpy as np, scipy.ndimage as ndi , PIL.Image as pil, cv2, argparse
from pathlib import Path



# In[53]:python .\CV-CellSeg2.0.py -img "C:\Users\mymmp\OneDrive\Notes\OBI\CV-Images\Images_Data\DL_CellSeg\Mouse_DAPI" -o "\Mouse_CV-CellSeg2" -wb1 True -wb2 True -we True -wm True
def break_1to_xy(file_path, out_dir, target_x = 512, target_y = 512):
    if file_path.is_file() and file_path.suffix=='.tif':   # if only a file
        img_name = file_path.stem
        im = skimage.io.imread(file_path)
        print("Original shape:", im.shape)
        im = skimage.exposure.rescale_intensity(im)
        try:
            w, h, c = im.shape
            is_gray = False
        except ValueError:   # grayscale image
            w, h = im.shape
            is_gray = True
        if w< target_x or h<target_y:
            raise Exception("This is not possible, target dimension below that of the original image.")
        num_x = w//target_x; start_x = (w%target_x)//2  # Left over room divided by 2 (centered)
        num_y = h//target_y; start_y = (h%target_y)//2

        if not out_dir.exists():
            os.mkdir(out_dir)
        else:
            print('Directory already exists, data maybe overwritten')

        with open(out_dir/"image_list.csv","a+") as f:
            [f.write(img_name +"_tile-x"+str(n)+"y"+str(m)+',\n')
                for n in range(num_x) for m in range(num_y)]

        if is_gray:
            # Make folders
            for m in range(num_y):
                for n in range(num_x):
                    img_dir = out_dir/(img_name + "_tile-x" + str(n) + "y" + str(m))
                    if not img_dir.exists():
                        os.mkdir(img_dir)
                        os.mkdir(img_dir/'images')

            # Save Images
            [skimage.io.imsave(img_dir/'images'/(img_name + "_tile-x" + str(n) + "y" + str(m)+".png"), 
                im[start_x+target_x*n:start_x+target_x*(n+1), start_y+target_y*m:start_y+target_y*(m+1)], check_contrast=0)
                for n in range(num_x) for m in range(num_y)]

    elif file_path.is_dir():  #is directory, reuse upper code
        for file in os.listdir(file_path):
            if file.endswith('.tif') or file.endswith('.png'):
                break_1to_xy(file_path/file, out_dir, target_x, target_y)

def CellSeg_Pipeline2(in_filename=Path("Blah.tif"),out_dir = Path("Output/")):
    # Make subfolders
    if not (out_dir/"images").exists():
        os.mkdir(out_dir/"images"); 
    else:
        print('Folder already exists, data maybe overwritten')
    #os.mkdir(out_dir/"masks")
    # Read
    img = cv2.imread(str(in_filename)); print('stack image shape:',img.shape)
    # write original image to out folder

    pil.fromarray(img).save(out_dir/"images"/Path(in_filename).name)



# In[37]:
if __name__ == "__main__":

    ap = argparse.ArgumentParser(prog='Traditional Cell Segmentation, input image, output mask', add_help=True)
    ap.add_argument("-img","--img", help= "Path to single image, or directory to images", required=True, type = str)
    ap.add_argument('-x',help='target x-dimension or (in num_mode) breaks in x wanted', default=1366, type=int)
    ap.add_argument('-y', help='target y-dimension or (in num_mode) breaks in y wanted', default=1366, type=int)
    
    args = ap.parse_args()
    
    crop_dir = Path(args.img)/'crops'; print("Crops output folder:", crop_dir)
    break_1to_xy(Path(args.img), out_dir = crop_dir , target_x = args.x, target_y = args.y)
    
    root_dir = Path(args.img)
    SEG_DIR = root_dir/"MRCNN-output"; print("Processed output folder:",SEG_DIR)

    if not SEG_DIR.exists():   # Mandatory output direcotry
        os.mkdir(SEG_DIR)
    else:
        print("Output folder already exist, data maybe overwritten")

    print("Preparation is completed.")

