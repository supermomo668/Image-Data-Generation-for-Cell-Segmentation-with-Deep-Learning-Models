#############################################################################
Normal Mode:
#############################################################################
Step 1. CellSeg_CV: Process a large amount of image and output their watershed segmentation image as baseline label:
	- Output: 
		1. Image and Masks
	- Used in next step:
		1. Image: the DAPI Image
		2. Masks: Boundary_Overlay.png

Step 2. ImageBreaker: Break a large image (e.g. 2048 x 2048) into smaller images ((512 x 512) x 16) for faster training, smaller data input size and allow manual removal of unwanted  areas later
	-Output: 
		1. DAPI Image(.png) in "n by n" dimensions
		2. Boundary_Overlay.png in "n by n" dimensions
	-Used in next step:
		1. "n by n" or 'tiles" of Boundary_Overlay.png

Step 3. Cell_Cropper (Bottleneck manual step): Edit the mask label of the individual tiles to correct/add/update masks from watershed segmentation
	- Output (all still "n by n"):
		1. Binary Contour
		2. Contour Overlay
		3. Crop Coordinates
	-Used in next step:
		1. Binary Contour - "Train Y"

Step 4. Pick and organized for training: 
	- Do:
		1. Eliminate FOVs (Field of Views) with too little feature as it adds noise
		2. Check/QC overall quality
		3. Move each (1. Base Image "Train X" & Label Mask Image "Train Y") into a single subdirectory in either "Training Data" or "Testing Data"
	- Output:
		1. DAPI Image(.png) in "n by n" dimensions (from step 2) - "Train X"
		2. Binary Contour - "Train Y"
		(Each example should be in respective subdirectory)

#############################################################################
Mask-RCNN Mode:
#############################################################################
Run 1.break images into 512 x 512 first
Then run 2.Cellseg-CV to generate mask and convert tiles images into the correct format
---------------------------
Use Batch script (or cmd) to execute those command inside the directory