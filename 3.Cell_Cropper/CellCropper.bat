SET CellCropper_PATH=C:\Users\mymmp\OneDrive\Notes\OBI\CV-Images\Image Functions\Cell_Cropper\Cell_Cropper.py

SET Input_PATH=C:\Users\mymmp\OneDrive\Notes\OBI\CV-Images\Image Functions\ImageBreaker\tiles_output

SET Output_PATH=C:\Users\mymmp\OneDrive\Notes\OBI\CV-Images\Image Functions\Cell_Cropper

:: Modify these
SET FOV_Name=BC68
SET tile_x=2
SET tile_y=1
      
python "%CellCropper_PATH%" -i "%Input_PATH%\%FOV_Name%_Boundary-overlay\boundary_overlay_tile-x%tile_x%y%tile_y%.png" -o "%Output_PATH%" -n "%FOV_Name%"
pause