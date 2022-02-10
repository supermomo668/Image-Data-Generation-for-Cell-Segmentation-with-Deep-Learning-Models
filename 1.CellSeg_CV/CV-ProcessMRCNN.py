#!/usr/bin/env python
# coding: utf-8

# In[48]:

import os,  numpy as np, random, skimage.io, matplotlib.pyplot as plt
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
        if not (out_dir/img_name).exists():
            os.mkdir(out_dir/img_name)
        output_dir = out_dir/img_name
                
        with open(out_dir/"image_list.csv","a+") as f:
            [f.write("'"+img_name +"_tile-x"+str(n)+"y"+str(m)+"'"+',\n')
                for n in range(num_x) for m in range(num_y)]

        if is_gray:

            [skimage.io.imsave(output_dir/(img_name + "_tile-x" + str(n) + "y" + str(m)+".png"), 
                im[start_x+target_x*n:start_x+target_x*(n+1), start_y+target_y*m:start_y+target_y*(m+1)], check_contrast=0)
                for n in range(num_x) for m in range(num_y)]

        else:

            [skimage.io.imsave(output_dir/ (img_name + "_tile-x" + str(n) + "y" + str(m)+".png"), 
                im[start_x+target_x*n:start_x+target_x*(n+1), start_y+target_y*m:start_y+target_y*(m+1), :], check_contrast=0)
                for n in range(num_x) for m in range(num_y)]

    elif file_path.is_dir():  #is directory, reuse upper code
        for file in os.listdir(file_path):
            if file.endswith('.tif') or file.endswith('.png'):
                break_1to_xy(file_path/file, out_dir, target_x, target_y)


def CellSeg_Pipeline2(in_filename=Path("Blah.tif"),out_dir = Path("Output/"), configs=configs,give_bounds=True, \
                      give_box=True, give_singlemask=True, give_allmasks=True, give_edges=True, just_mask = False, remove_bordering=False):
    # Make subfolders
    os.mkdir(out_dir/"images"); 
    #os.mkdir(out_dir/"masks")
    # Read
    img = cv2.imread(str(in_filename)); print('stack image shape:',img.shape)
    # write original image to out folder

    pil.fromarray(img).save(out_dir/"images"/Path(in_filename).name)



# In[37]:
if __name__ == "__main__":

    ap = argparse.ArgumentParser(prog='MRCNN Cell Segmentation Preparation', add_help=True)
    ap.add_argument("-img","--image", help= "Path to single image, or directory to images", required=True, type = str)
    ap.add_argument("-o","--out_dir", help= "Name of Output directory",type=str)

    ap.add_argument('-x',help='target x-dimension or (in num_mode) breaks in x wanted', required=True, type=int)
    ap.add_argument('-y', help='target y-dimension or (in num_mode) breaks in y wanted', required=True, type=int)

    args = ap.parse_args()

    crops_out_dir = Path(args.img/'Crops')

    break_1to_xy(Path(args.img), out_dir = crops_out_dir,\
        target_x = args.x, target_y = args.y)
    
    CWD = Path(os.getcwd())
    IMG_DIR = crops_out_dir; print("Input directory or path (from crop):",IMG_DIR)
    OUT_DIR = Path(IMG_DIR/"MRCNN-ready"); print("Output folder name:",OUT_DIR)
    
    if not OUT_DIR.exists():   # Mandatory output direcotry
        os.mkdir(OUT_DIR)
    else:
        print("Output folder already exist, data maybe overwritten.")

    if not OUT_DIR.exists():
        os.mkdir(OUT_DIR)
    else:
        print("MRCNN output folder already exist, data maybe overwritten.")
        
    for root, dirs, files in os.walk(crops_out_dir):
        for crop_img in dirs:
            sub_img_dir = Path(os.join(root, dirs))
            print("Processing file at", sub_img_dir)
            img_name = sub_img_dir/crop_img
            if img_name.endswith(".tif") or img_name.endswith(".png"):
                # Mask Subfolder
                out_dir = OUT_DIR/img_name.stem
                if out_dir.exists():
                    print("Case folder already exist, data maybe overwritten or os error")
                else:
                    os.mkdir(out_dir)

                CellSeg_Pipeline2(in_filename=img_name, out_dir=out_dir)
            else:
                print("File doesn't seem to be in the right format.")


print("Pipeline is completed! Congratulations!")