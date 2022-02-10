#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np, os, skimage.io, skimage.exposure, argparse, pathlib


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
            [f.write(img_name +"_tile-x"+str(n)+"y"+str(m)+',\n')
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
               
if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog='Break Images into tiles')

    ap.add_argument('-img', help='image directory or name', required =True)
    ap.add_argument('-x',help='target x-dimension or (in num_mode) breaks in x wanted', required=True, type=int)
    ap.add_argument('-y', help='target y-dimension or (in num_mode) breaks in y wanted', required=True, type=int)
    ap.add_argument('-o', '--out_dir', help='Output directory name',default="crops",type=str)

    args = ap.parse_args()

    break_1to_xy(pathlib.Path(args.img), out_dir = pathlib.Path(args.img)/'crops',\
        target_x = args.x, target_y = args.y)
