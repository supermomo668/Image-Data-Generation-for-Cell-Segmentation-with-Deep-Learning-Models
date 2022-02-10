set image_name=BC68_DAPI

conda activate imgdl & python Image_Breaker.py -img "C:\Users\mymmp\OneDrive\Notes\OBI\CV-Images\Image Functions\CellSeg_CV\Mouse_CV-CellSeg\%image_name%\image\%image_name%.png" -x 512 -y 512 -oc "%image_name%"
pause