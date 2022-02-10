SET FOV_Name=BU29_DAPI
SET CellSeg_PATH=C:\Users\mymmp\OneDrive\Notes\OBI\CV-Images\Image Functions\CellSeg_CV\Mouse_CV-CellSeg

conda activate imgdl & python Image_Breaker.py -img "%CellSeg_PATH%\%FOV_Name%\masks\boundary_overlay.png" -x 512 -y 512 -oc "%FOV_NAME:~0,4%_Boundary-overlay"
pause