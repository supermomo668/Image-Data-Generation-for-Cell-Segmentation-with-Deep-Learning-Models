import numpy as np, os, skimage.io, skimage.exposure, argparse
for filename in os.listdir:
	print(filename)

image_name = "AV81_DAPI.tif"

def break_1toN(file_path, break_x=2, break_y=2):  # break 1 image to 4
	img_name = file_path.split()
	im = skimage.io.imread(file_path)
	im = skimage.exposure.rescale_intensity(im)
	w, h = im.shape
	print("Original shape", w, h)
	M = w//break_x
	N = h//break_y
	[skimage.io.imsave(image_name++"_x"+str(x)+'_'+str(y), 
		im[x:x+M,y:y+N]) for x in range(0,w, M) 
		for y in range(0,h,N)]

	

if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_help("Break images into tiles")
	ap.add_argument('-img', 'image directory or name')
	ap.add_argument('-x',help='target x-dimension', type=int)
	ap.add_argument('-y', help='target y-dimension',type=int)
	args  = ap.parse_args()

	break_1toN(args.img, break_x = arg.x, break_y = arg.y)
