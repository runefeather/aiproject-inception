from PIL import Image
import os
import numpy as np

# # Get the mean of the images
def getMean(path_to_images): # ../testing/..
	"""prec
		return: mean value
	"""
	meanList = list()
	arr = list()

	# mean for all images
	for root, dirs, files in os.walk(path_to_images):
		for f in files:
			img = Image.open(os.path.join(root, f))
			arr.append(np.mean(np.array(img), axis=(0,1)))
	for i in arr:
		print(i)
	# print(np.mean(arr, axis=(0,1)))
	return arr



# get the mean for single image
def getMean(file):
	img = Image.open(file)
	arr = np.array(img)
	return np.mean(arr, axis=(0, 1))


def downSampling(image, w, h):
	img = Image.open(image)

	if 'P' in img.mode:
		img = img.convert("RGB")
		img = img.resize((w, h), Image.ANTIALIAS)
		img = img.convert("P", dither=Image.NONE, palette=Image.ADAPTIVE)
	else:
		img = img.resize((w, h), Image.ANTIALIAS)

	return img


# this function perform sliding on an input image
# output is an array of numpy arrays
def sliding(img):
	
	WINDOW = 64
	arr = []

	width, height = img.size
	# print(width, height)

	# if img is not 64x64
	if(height < 64 or width < 64):
		return [np.array(img)]

	# Eg: img 240 x 240, slider WINDOW x WINDOW
	if(width % WINDOW == 0):
		numStepsHorizontal = width // WINDOW
	else:
		numStepsHorizontal = (width // WINDOW) + 1
	
	if(height % WINDOW == 0):
		numStepsVertical = height // WINDOW
	else:
		numStepsVertical = (height // WINDOW) + 1

	# print("moveHorizontal: ", numStepsHorizontal)
	# print("moveVertical: ", numStepsVertical)

	tempHorizontal = WINDOW - (width//numStepsHorizontal) 
	tempVertical = WINDOW - (height//numStepsVertical)

	sizeStepHorizontal = WINDOW - tempHorizontal
	sizeStepVertical = WINDOW - tempVertical

	# print("sizeStepHorizontal: ", sizeStepHorizontal)
	# print("sizeStepVertical: ", sizeStepVertical)

	w = 0
	h = 0
	for i in range(0, width, sizeStepHorizontal):
		for j in range(0, height, sizeStepVertical):
			cr = img.crop((i, j, i+WINDOW, j+WINDOW))
			a = np.array(cr)
			arr.append(a)

	# print(len(arr))
	return arr


# if __name__ == '__main__':
# 	img = "/home/runefeather/Pictures/16807397_1161214887323763_4932185148616037617_n.png"
# 	a = sliding(img)
# 	newimg = Image.fromarray(a[5])
# 	newimg.save("img1.png")


