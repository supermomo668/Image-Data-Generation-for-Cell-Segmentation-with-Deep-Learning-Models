#!/usr/bin/env python
# coding: utf-8

import numpy as np, cv2, matplotlib.pyplot as plt, os.path, mahotas as mh
import argparse, pathlib

# In[6]:

def cell_crop(event, x, y, flags, param):

    global clone, show_clone, crop_num, all_segments, new_crop, cropping, this_crop, last_img, cropping, accept
    global draw_mode, erase_mode, continuous_drawerase, previous_point, mask_clone
    eraser_radius = 5
    
    if not cropping and accept:
        if not draw_mode:
            clone = cv2.drawContours(last_img, [this_crop], -1, (0, 255,0), 1, cv2.LINE_AA)
            show_clone = clone.copy()
        else:
            clone = show_clone
        print("Segment accepted")
        cv2.imshow("Cell Cropper", clone)
        last_img = clone.copy()
        accept = False
        
    elif event == cv2.EVENT_LBUTTONDOWN:
        if not erase_mode:
            if not cropping and new_crop:
                last_img = clone.copy()
                this_crop = np.empty((0,2),dtype = int)
                new_crop = False
                cropping = True
                accept = False
                if draw_mode:
                    crop_num += 1; print('Crop', crop_num,". Drawing ...")
                    show_clone = clone.copy()
                    continuous_drawerase = True
                    previous_point = (x,y)
                    this_crop = np.append(this_crop, np.array([(x,y)]), axis = 0)
                else:
                    crop_num += 1; print('Crop', crop_num)
                    
            if not draw_mode:
                this_crop = np.append(this_crop, np.array([[x, y]]), axis = 0)
                clone = show_clone
                cv2.imshow("Cell Cropper", show_clone)
            
        elif erase_mode:
            last_img = clone.copy()
            mask_clone = clone[:,:,1].copy()
            new_crop = False
            cropping = True
            print("Erasing...")
            cv2.circle(mask_clone,(x,y),10,(0,0,0),10)
            cv2.imshow("Cell Cropper", clone)
            continuous_drawerase = True
            
    elif event == cv2.EVENT_LBUTTONUP:
        continuous_drawerase = False
        if draw_mode:
            print("Draw stopped.")
            cropping = False
        elif erase_mode:
            cropping = False
            print("Erase stopped.")
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping and not new_crop: 
            if not continuous_drawerase:
                show_clone = clone.copy()
                cv2.line(show_clone,(x,y),tuple(this_crop[-1]),(0,0,255),4) # draw line between former and present pixel
                cv2.imshow("Cell Cropper", show_clone)    # draw line between former and present pixel
                
            if continuous_drawerase:   # When user is holding click button to select boundary
                if draw_mode:
                    cv2.line(show_clone,(x,y), previous_point, (0, 255, 0), 2)
                    this_crop = np.append(this_crop, np.array([(x,y)]), axis = 0)
                    previous_point = (x,y)
                    cv2.imshow("Cell Cropper", show_clone)
                elif erase_mode:
                    cv2.circle(mask_clone,(x,y),eraser_radius,(0,0,0),eraser_radius)
                    clone[:,:,1] = mask_clone.copy()
                    cv2.imshow("Cell Cropper", clone)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--image", help="Input image directory",required=True)
    ap.add_argument("-o","--out_dir", help="Location of output directory (we will make a folder for you)", \
        default=os.getcwd(), type=str)
    ap.add_argument("-n", "--name", help="Name of output SUB-directory (will create one)", type=str)
    args = ap.parse_args()

    CWD = pathlib.Path(os.getcwd())
    img_path = pathlib.Path(args.image)
    case_name = img_path.stem
    Output_dir =  pathlib.Path(args.out_dir)/"CellCropper_Output"
    if not Output_dir.exists():     # Make Outermost Output directory
        os.makedirs(Output_dir)
        
    if args.name :
	    out_dir = Output_dir/args.name
    else:
	    out_dir = Output_dir
    print("Selected image at:", img_path,"\nOutput directory at", args.out_dir)

    img = cv2.imread(str(img_path))
    print("Input image shape", img.shape)
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

    clone = img.copy(); 
    all_segments = []
    this_crop = np.empty((0,2),dtype = int)
    new_crop = True
    cropping = False
    accept = False
    crop_num = 0 
    draw_mode = False
    erase_mode = False
    continuous_drawerase = False
    
    cv2.namedWindow("Cell Cropper", cv2.WINDOW_GUI_NORMAL|cv2.WND_PROP_ASPECT_RATIO); 
    cv2.setMouseCallback("Cell Cropper", cell_crop)
    cv2.imshow("Cell Cropper", clone)
    show_clone = clone.copy()
    
    # keep looping until the 'q' key is pressed
    while True:
    # display the image and wait for a keypress
        key = cv2.waitKey() #1) & 0xFF
        if key == ord("r"):
            print("Reset to previous operation.")
            clone = last_img
            show_clone = last_img
            new_crop = True
            cropping = False
            if not erase_mode:
                crop_num -= 1

        elif key == ord("a"):  # accept
            all_segments.append(this_crop)
            new_crop = True
            cropping = False
            accept = True

        elif key == ord("e"):
            if erase_mode:
                erase_mode = False
                cropping = False
                new_crop = True
                print("Erase mode: Completed & Off")
                last_img = clone.copy()
                show_clone = clone.copy()
            else:   # Erase mode on
                erase_mode = True
                draw_mode = False
                print("Erase mode: On")

        elif key == ord("c"):
            if draw_mode:
                draw_mode = False
                print("Free draw mode: Off")
            else:
                draw_mode = True
                erase_mode = False
                print("Free draw mode: On")

        elif key == ord("s") and not cropping:   # save
            if not out_dir.exists():     # Make output directory & subdirectories
                os.makedirs(out_dir)
            # Subdirectories
            output_dir_names = ["Crop Coordinates", "Bin-Contour","Contour_Overlay"]
            for name in output_dir_names:
                if not (out_dir/name).exists():
                    os.makedirs(out_dir/name)
            
            # Skip if nothing is cropped
            if len(all_segments) != 0:
                fid = 0
                # 1. Coordinate path 2. Binary Mask Path 3. Binary overlay path
                output_3paths = [out_dir/output_dir_names[0]/(case_name+'_'+str(fid)+'.csv'),
                                out_dir/output_dir_names[1]/(case_name+'_'+str(fid)+".png"),
                                out_dir/output_dir_names[2]/(case_name+'_'+str(fid)+".png")]
                print("Output subdirectories created")
                # In case of duplicate
                while output_3paths[0].exists() or output_3paths[1].exists() or output_3paths[2].exists():
                    fid += 1
                    output_3paths = [out_dir/output_dir_names[0]/(case_name+'_'+str(fid)+'.csv'),
                            out_dir/output_dir_names[1]/(case_name+'_'+str(fid)+".png"),
                            out_dir/output_dir_names[2]/(case_name+'_'+str(fid)+".png")]
                    print("File duplicate found, file ID changed to ", fid)


                # write coordinate file
                with open(output_3paths[0], 'w', newline = '') as f_out:
                    for seg in all_segments:
                        for i in range(len(seg)):
                            f_out.write(str(seg[i][0])+','+str(seg[i][1])+'\t')
                        f_out.write('\n')
                # Save contour mask & overlay
                bin_contour = (clone[:,:,1]==255).astype('uint8')*255
                cv2.imwrite(str(output_3paths[1]), bin_contour)
                overlay = mh.as_rgb(clone[:,:,0], bin_contour, clone[:,:,2])
                cv2.imwrite(str(output_3paths[2]), overlay)

                print("Coordinates are saved to:", output_3paths[0])
                print("Binary Mask is saved to", output_3paths[1])
                print("Mask Overlay is saved to", output_3paths[2])
            else:
                print("File not saved as no new segments made")
        
        elif key == 27 or key == ord("q"):  # Quit
            new_crop = True
            cropping = False
            accept = True
            break
        cv2.imshow("Cell Cropper", clone)

    cv2.destroyAllWindows()


