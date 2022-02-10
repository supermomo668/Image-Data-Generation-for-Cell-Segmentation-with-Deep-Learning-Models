conda activate imgdl
set image_name=HumanTissue\AV69_DAPI

python Image_Breaker.py -img C:\Users\mymmp\OneDrive\Work_Notes\OBI\CV-Images\ImageProcessing\2.ImageBreaker\Original_Images\%image_name%.tif -x 512 -y 512
pause