from PIL import Image
import os
import numpy as np

# # Get the mean of the images
def getMean(path_to_images): # ../testing/..
	"""
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
