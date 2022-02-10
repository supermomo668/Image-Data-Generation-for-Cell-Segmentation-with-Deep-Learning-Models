#!/usr/bin/env python
# coding: utf-8

# In[48]:

import mahotas as mh, os,  numpy as np, random, skimage.io, matplotlib.pyplot as plt
from skimage.filters import rank,  threshold_local, threshold_otsu
from skimage.morphology import disk
from skimage.feature import peak_local_max
import numpy as np, scipy.ndimage as ndi , PIL.Image as pil, cv2
import cv2 
import argparse
from pathlib import Path
import multiprocessing
import matplotlib
matplotlib.use('TkAgg')


# In[53]:python .\CV-CellSeg2.0.py -img "C:\Users\mymmp\OneDrive\Notes\OBI\CV-Images\Images_Data\DL_CellSeg\Mouse_DAPI" -o "\Mouse_CV-CellSeg2" -wb1 True -wb2 True -we True -wm True

# Configs (default)
configs = {'cellsize_LowLim_filter':16**2,
           'blur_avgcellsize':16,
           'blurg_sigma':0.5,
           'local_threshold_blocksize':25}
# In[77]:

def Blurg_maximas(target, blur_factor=16, sigma=0.5, dil4show = 15):   

    # Apply Gaussian-mean mask
    gaussian_f = mh.gaussian_filter(target, sigma = sigma); g_fitler = target>gaussian_f.mean()
    #ax1 = fig.add_subplot(131); ax1.imshow(target*g_fitler); ax1.set_title('gaussian')
    
    # Blur resultant image
    img_blurred= rank.mean(target, disk(blur_factor)); 
    gaussian_f = mh.gaussian_filter(img_blurred, sigma = sigma); 
    blurg_filter = gaussian_f>gaussian_f.mean(); filtered_img = img_blurred*blurg_filter
    
    # Maximas
    maxima = mh.regmax(filtered_img,); 
    maxima_labeled, _ = mh.label(maxima)
    #ax2 = fig.add_subplot(133); ax2.imshow(filtered_img); ax2.set_title('Blur then Gaussian (filtered_img)')
    return g_fitler, img_blurred, blurg_filter, filtered_img, maxima_labeled

# In[60]:

from skimage.measure import regionprops
import matplotlib.patches as mpatches

# QC Functions: methods
''' (QC Use)
def Label_sampler(labeled_objects, xy = 8, num_labels = 100):
    fig = plt.figure(figsize=(5*xy,5*xy)); start = random.randint(1, num_labels-xy**2)
    for n, i in enumerate(range(start, start+xy**2)):
        ax = fig.add_subplot(xy, xy,n+1)
        ax.imshow(labeled_objects==i)
'''
# Filter Watershed label by size
def Label_boxer(labeled_objects, min_size = 20, max_size = None, \
    area_record = False, savefig=True, figname="BoxLabels.png"):
    fig, ax = plt.subplots(1,1)
    box_coords = {}; area_records = {}
    if not max_size:
        max_size = np.inf
    for region in regionprops(labeled_objects):
        area = region.area; label = region.label
        if min_size > area or area > max_size:
            continue
        minr, minc, maxr, maxc = region.bbox
        box_coords[label] = (minr, minc, maxr, maxc)   # Record the region
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=0.7)
        ax.add_patch(rect)
        area_records[label] = area
    if area_record:
        return box_coords, area_records
    else:
        return box_coords
    if savefig:
        ax.savefig(figname)

# In[65]:

#  Watershed
from skimage.morphology import watershed, disk

def Segmentation_by_maxi(mask, maxima_labeled, footprint = 8):
    # Generate maximas
    distance = ndi.distance_transform_edt(mask)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((footprint, footprint)), labels=mask, 
                               threshold_rel=10)
    markers = ndi.label(maxima_labeled)[0];    # Your maximas here
    #markers = maxima_labeled
    labels = watershed(-distance, markers, mask=mask)
    print('Number of objects detected: ', markers.max())
    
    return labels, local_maxi

# In[67]:


def size_filter(watershed_labels, min_size = 16**2):
    print("filtering size of {}...".format(min_size))
    sizes = mh.labeled.labeled_size(watershed_labels)
    labels = mh.labeled.remove_regions_where(watershed_labels, sizes < min_size)
    print('Number of objects after filter',len(np.unique(labels)))
    return labels


# In[35]:
#  Output Segmentation
from skimage.segmentation import find_boundaries
            
def Labeled2Nuclear(img, cell_labels, cell_boxes, method = 'Bin_Thres', local_contrast_radius = 2, output_coords = False):
    img_native = img.copy(); output_coords = {}

    if method == 'Bin_Thres':
        bin_nuclear_img = np.full(img.shape, False, dtype= bool)
        for label, box in cell_boxes.items(): # Cell
            (x_min, y_min, x_max, y_max) = box; img_ex = img[x_min:x_max, y_min: y_max]  # Made an image out of bbox
            # Local contrast and bin threshold
            enh_ex = mh.stretch(img_ex, disk(local_contrast_radius))
            hist_intensity = np.histogram(enh_ex); thres_bin = hist_intensity[1][-2]
            coords = np.where(enh_ex > thres_bin);   # print('bbox nuclear', coords, 'bbox', box)
            nuclear_real_coordinates = np.add(coords, np.array([[x_min],[y_min]]))
            bin_nuclear_img[nuclear_real_coordinates[0],nuclear_real_coordinates[1]] = True
        return bin_nuclear_img

    elif method == 'Local_boundary':
        bin_boundary_img = np.full(img.shape, False, dtype= bool)
        for label, box in cell_boxes.items(): # Cell
            (x_min, y_min, x_max, y_max) = box; label_ex = cell_labels[x_min:x_max, y_min: y_max]  # Made an image out of bbox
            boundaries_bin = find_boundaries(label_ex, connectivity=1, mode='thick', background=0)
            coords = np.where(boundaries_bin ==True)
            real_bound_coords = np.add(coords, np.array([[x_min],[y_min]]))
            bin_boundary_img[real_bound_coords[0], real_bound_coords[1]] = True
        return bin_boundary_img

    elif method == 'Show_boundary':
        boundaries_bin = find_boundaries(cell_labels, connectivity=1, mode='thick', background=0)
        return boundaries_bin

# Pipeline function
def CellSeg_Pipeline(in_filename=Path("Blah.tif"),out_dir = Path("Output/"), configs=configs, give_bounds=True, \
                     give_box=True, give_singlemask=True, give_allmasks=True, give_edges=True, just_mask = False, remove_bordering=False):
    # Make subfolders
    os.mkdir(out_dir/"images")
    os.mkdir(out_dir/"masks")
    # Read
    dna = mh.stretch(skimage.io.imread(in_filename, as_gray=True))
    print('stack image shape:',dna.shape)
        # write original image to out folder
    if just_mask:
        pil.fromarray(dna).save(out_dir/"images"/Path(in_filename).name)
    else:
        pil.fromarray(dna).save(out_dir/"images"/("DAPI.png"))

    # Gaussian blurring filter
    g_img = mh.gaussian_filter(dna, 3); otsu = mh.otsu(dna);
    bin_image= g_img > otsu
    #local_img = threshold_local(dna, block_size=local_threshold_blocksize)
    #bin_image = local_img > mh.otsu(mh.stretch(local_img))
    
    # label the image in order to assign index to each component
    labeled, nr_object = mh.label(bin_image)
    #target = bin_image*dna
    target = dna
    print('Initial no. of objects:',nr_object)

    _,_, blurg_filter, filtered_img, maxima_labeled = Blurg_maximas(target, 
                                                                    blur_factor=configs['blur_avgcellsize'], sigma=configs['blurg_sigma'])
    mask = blurg_filter
    watershed_filter2, local_maxi = Segmentation_by_maxi(mask, maxima_labeled, footprint=6)
    watershed_labels = watershed_filter2
    
    if remove_bordering:
        # ### Cleaning up regions , remove cells touching the border (Optional)
        watershed_labels = mh.labeled.remove_bordering(watershed_labels)
        print('Remove bordering...No. of objects:', len(np.unique(watershed_labels)))

    #remove things that are too small to be a cell
    final_labels = size_filter(watershed_labels, min_size = configs['cellsize_LowLim_filter'])
    final_num_objs = len(np.unique(final_labels))
    print('Remove Small objects...final number of objects:',final_num_objs)
    
    # Filter with original threshold binary to reduce watershed boundaries 
    final_labels *= bin_image

# Outputs
    print("Output results...")
    #   Labels
    if not just_mask:
        pil.fromarray(final_labels).save(out_dir/"masks"/"labels.png")
    # Boxes (with img)
    if give_box == True:
        print("1. Box-overlay image")
        boxes = Label_boxer(final_labels, min_size= configs['cellsize_LowLim_filter'], savefig=True,\
                            figname=OUT_DIR/"masks"/"BoxLabels.png")
    #   Binary Mask
    if give_singlemask:
        print("2a. Binary mask of all non-background region ")
        pil.fromarray(final_labels!=0).save(out_dir/"masks"/"mask.png")
        
    if give_allmasks:
        print("2b. Binary mask of each labeled region (including background)")
        if not just_mask:
            os.mkdir(out_dir/"masks"/"each")
            # Background
            pil.fromarray(final_labels==0).save(out_dir/"masks"/"each"/"background.png")
            for i in range(1,final_labels.max()):
                pil.fromarray(final_labels==i).save(out_dir/"masks"/"each"/("mask"+str(i)+".png"))
        else:
            for i in range(1,final_labels.max()):
                pil.fromarray(final_labels==i).save(out_dir/"masks"/("mask"+str(i)+".png"))
        
    #   Binary Boundary
    if give_bounds:
        print("3. Binary watershed boundary")
        bin_boundary_img = Labeled2Nuclear(img = filtered_img, cell_labels = final_labels, cell_boxes = boxes, method = 'Show_boundary')
        pil.fromarray(bin_boundary_img).save(out_dir/"masks"/"boundary.png")
        # Write overlay
        overlay = mh.as_rgb(dna, bin_boundary_img, dna)
        pil.fromarray(overlay).save(out_dir/"masks"/"boundary_overlay.png")

    #   edge detection Canny, histogram binning threshold
    if give_edges:
        print("4. Binary edges (with auto-binning)")
        h = plt.hist(np.hstack(filtered_img), bins=4)
        lowlim_hysteresis, highlim_hysteresis = h[1][1], h[1][2]
        bin_edges = cv2.Canny(filtered_img, lowlim_hysteresis, highlim_hysteresis)
        overlay = mh.as_rgb(dna, bin_edges, dna)
        pil.fromarray(overlay).save(out_dir/"masks"/"edge_overlay.png")

# mode 1   
def CellSeg_Pipeline2(in_filename=Path("Blah.tif"),out_dir = Path("Output/"), configs=configs,give_bounds=True, \
                      give_box=True, give_singlemask=True, give_allmasks=True, give_edges=True, just_mask = False, remove_bordering=False):
    # Make subfolders
    os.mkdir(out_dir/"images"); os.mkdir(out_dir/"masks")
    # Read
    img = cv2.imread(str(in_filename)); print('stack image shape:',img.shape)
       # write original image to out folder
    if just_mask:
        pil.fromarray(img).save(out_dir/"images"/Path(in_filename).name)
    else:
        pil.fromarray(img).save(out_dir/"images"/("DAPI.png"))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adapt_thresh = (((threshold_local(gray, block_size=5) > threshold_otsu(gray))!=0)*255).astype('uint8')

    opening = adapt_thresh

    # noise removal, sure background area
    kernel = np.ones((configs['kernel_size'], configs['kernel_size']), np.uint8)
    sure_bg = cv2.dilate(opening, kernel, iterations=configs['dilate_iteration'])

    # Finding sure foreground area  
    dist_transform = mh.stretch(cv2.distanceTransform(opening, cv2.DIST_L2, 0))
    sure_fg = ((threshold_local(dist_transform, block_size=configs['adapt_block_size'], offset=configs['adapt_offset'], method='gaussian', mode='mirror') > dist_transform.mean())!= 0)*255
    #ret, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg); unknown = cv2.subtract(sure_bg, sure_fg)
    
    ret, markers = cv2.connectedComponents(sure_fg)
    # background is  0, boundary is saved
    markers = markers+1; markers[unknown==255] = 0
    watershed_labels = cv2.watershed(img, markers)
    bin_boundary_img = watershed_labels == -1;  # Save  bounds
    watershed_labels -= 1; watershed_labels[watershed_labels<0] = 0   # Remove negative to do size filtering
    
    '''
    #remove things that are too small to be a cell
    final_labels = size_filter(watershed_labels, min_size = configs['cellsize_LowLim_filter'])
    final_num_objs = len(np.unique(final_labels))
    print('Remove Small objects...final number of objects:', final_num_objs)
    '''
    # Filter with original threshold binary to reduce watershed boundaries 
    final_labels = watershed_labels


# Outputs
    print("Output results...")
    #   Labels
    if not just_mask:
        pil.fromarray(final_labels).save(out_dir/"masks"/"labels.png")
    # Boxes (with img)
    if give_box == True:
        print("1. Box-overlay image")
        boxes = Label_boxer(final_labels, min_size= configs['cellsize_LowLim_filter'], savefig=True,\
                            figname=OUT_DIR/"masks"/"BoxLabels.png")
    #   Binary Mask
    if give_singlemask:
        print("2a. Binary mask of all non-background region")
        pil.fromarray(final_labels!=0).save(out_dir/"masks"/"mask.png")
        
    if give_allmasks:
        print("2b. Binary mask of each labeled region (including background)")
        if not just_mask:
            os.mkdir(out_dir/"masks"/"each")
            # Background
            pil.fromarray(final_labels==0).save(out_dir/"masks"/"each"/"background.png")
            for i in range(1,final_labels.max()):
                pil.fromarray(final_labels==i).save(out_dir/"masks"/"each"/("mask"+str(i)+".png"))
        else:
            pil.fromarray(final_labels!=0).save(out_dir/"masks"/("BG_mask0"+".png"))
            for i in range(1,final_labels.max()):
                pil.fromarray(final_labels==i).save(out_dir/"masks"/("mask"+str(i)+".png"))
        
    #   Binary Boundary
    if give_bounds:
        print("3. Binary watershed boundary")
        
        pil.fromarray(bin_boundary_img).save(out_dir/"masks"/"boundary.png")
        # Write overlay
        overlay = mh.as_rgb(gray, bin_boundary_img, gray)
        pil.fromarray(overlay).save(out_dir/"masks"/"boundary_overlay.png")

    #   edge detection Canny, histogram binning threshold
    if give_edges:
        print("4. Binary edges (with auto-binning)")
        h = plt.hist(np.hstack(gray), bins=4)
        lowlim_hysteresis, highlim_hysteresis = h[1][1], h[1][2]
        bin_edges = cv2.Canny(gray, lowlim_hysteresis, highlim_hysteresis)
        overlay = mh.as_rgb(gray, bin_edges, gray)
        pil.fromarray(overlay).save(out_dir/"masks"/"edge_overlay.png")



# In[37]:
if __name__ == "__main__":
    WORKING_DIR = os.getcwd()

    ap = argparse.ArgumentParser(prog='Traditional Cell Segmentation, input image, output mask', add_help=True)
    ap.add_argument("-img","--image", help= "Path to single image, or directory to images", required=True, type = str)
    ap.add_argument("-out","--out_dir", help= "Name of Output directory", type=str)
    ap.add_argument("-wb1", "--bin_bound", help="Output binary boundaries", default=True, type = bool)
    ap.add_argument("-wb2", "--bin_box", help="Output boxes overlay", default=True, type = bool)
    ap.add_argument("-we", "--bin_edge", help="Output edge & edge-image overlay", default=True, type = bool)
    ap.add_argument("-wsm", "--bin_singlemask", help="Output binary mask", default=True, type=bool)
    ap.add_argument("-wam","--bin_allmask", help="Output all individual mask", default=True, type=bool)
    ap.add_argument("-mode", "--mode", help="Segmentation Mode (0 or 1)", default=0, type=int)
    ap.add_argument("-just_mask","--just_mask", help="Output only binary masks in 'masks' directory for Mask R-CNN directory format", default=False)
    
    args = ap.parse_args()
    
    if args.just_mask:
        args.bin_bound = False
        args.bin_box = False
        args.bin_edge = False
        args.bin_singlemask = False
    
    # Input restrictions
    assert args.mode in [0, 1]
    
    # Configs (default)
    configs = {'mode': 0,
               'cellsize_LowLim_filter':16**2,
               'blur_avgcellsize':13,
               'blurg_sigma':0.5,
               'local_threshold_blocksize':25}
    
    if args.mode == 1:
        # for cells
        configs = {'mode':1,
                   'cellsize_LowLim_filter':16**2,
                   'kernel_size':3, 
                   'dilate_iteration':1, 
                   'dist_thre shold':0.7,
                   'dist_blur':3, 
                   'adapt_block_size':21,
                   'adapt_offset':100}
        
    print("Using the following configuration", configs)
    
    CWD = Path(os.getcwd())
    IMG_DIR = Path(args.image); print("Input directory or path:",IMG_DIR)
    OUT_DIR = Path(CWD/"results"); print("Output folder name:",OUT_DIR)
    
    if not Path(CWD/"results").exists():   # Mandatory output direcotry
        os.mkdir(CWD/"results")
    else:
        print("Output folder already exist, data maybe overwritten")

    if IMG_DIR.is_dir(): # Images in a directory
        if not args.out_dir :   # if out_dir name is given:
            OUT_DIR = OUT_DIR/IMG_DIR.stem
        else:
            OUT_DIR = OUT_DIR/args.out_dir
        if not OUT_DIR.exists():
            os.mkdir(OUT_DIR)
        for file in os.listdir(IMG_DIR):
            file_dir = IMG_DIR/file
            print("Processing file at", file_dir)
            if file.endswith(".tif") or file.endswith(".png"):
                # Mask Subfolder
                out_dir = OUT_DIR/file_dir.stem
                if out_dir.exists():
                    print("Case folder already exist, data maybe overwritten or os error")
                else:
                    os.mkdir(out_dir)
                if args.mode==0:
                    print("Running mode 0 for full-size image")
                    CellSeg_Pipeline(in_filename=IMG_DIR/file, out_dir=out_dir, configs=configs, give_bounds=args.bin_bound,\
                        give_box=args.bin_box, give_singlemask=args.bin_singlemask, give_allmasks=args.bin_allmask,\
                                    give_edges=args.bin_edge, just_mask=args.just_mask)            
                if args.mode==1:
                    print("Running mode 1 for MRCNN")
                    CellSeg_Pipeline2(in_filename=IMG_DIR/file, out_dir=out_dir, configs=configs, give_bounds=args.bin_bound,\
                        give_box=args.bin_box, give_singlemask=args.bin_singlemask, give_allmasks=args.bin_allmask, \
                                    give_edges=args.bin_edge, just_mask=args.just_mask)
            else:
                print("File doesn't seem to be in the right format.")
    elif str(IMG_DIR).endswith(".tif"):   # Just one image
        print("Running on single image.")
        if OUT_DIR.exists():
            print("Case folder already exist, data maybe overwritten")
        else:
            if args.mode==0:
                print("Running mode 0 for full-size image")
                CellSeg_Pipeline(in_filename=IMG_DIR, out_dir=OUT_DIR/IMG_DIR.stem)
            elif args.mode==1:
                print("Running mode 1 for MRCNN")
                CellSeg_Pipeline2(in_filename=IMG_DIR, out_dir=OUT_DIR/IMG_DIR.stem, configs=configs, give_bounds=args.bin_bound,\
                    give_box=args.bin_box, give_singlemask=args.bin_singlemask, give_allmasks=args.bin_allmask, \
                                give_edges=args.bin_edge, just_mask=args.just_mask)
    else:
        raise Exception("The directory does not exist or is the wrong format.")

print("Pipeline is completed! Congratulations!")