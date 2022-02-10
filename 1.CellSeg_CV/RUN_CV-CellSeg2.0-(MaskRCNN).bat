set tile_folder=AV69_DAPI
conda activate imgdl && python "CV-CellSeg2.0.py" -img C:\Users\mymmp\OneDrive\Work_Notes\OBI\CV-Images\ImageProcessing\2.ImageBreaker\results-tiles\%tile_folder% -out %tile_folder%-modeRCNN -wb1 True -wb2 True -we True -wsm True -wam True -mode 1 -just_mask True
pause