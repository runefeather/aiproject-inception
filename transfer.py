import os
from shutil import copyfile

def split(test, train, val, splitnum):
	testfile = open("/home/runefeather/Desktop/Classwork/AI/Project/inception/data/tumor/" + splitnum +"/testdata.txt", "w+")
	trainfile = open("/home/runefeather/Desktop/Classwork/AI/Project/inception/data/tumor/" + splitnum +"/traindata.txt", "w+")
	valfile = open("/home/runefeather/Desktop/Classwork/AI/Project/inception/data/tumor/" + splitnum +"/valdata.txt", "w+")	


	for files in test:
		with open(files) as f:
			for line in f:
				if splitnum in files:
					testfile.write(line)
					copyfile("/home/runefeather/Desktop/Classwork/AI/Project/BreaKHis_data/" +line.split(' ')[0], "/home/runefeather/Desktop/Classwork/AI/Project/inception/data/tumor/" + splitnum +"/testing/" + line.split(' ')[0].split('/')[-1])
	print("Testing done")
	for files in train:
		with open(files) as f:
			for line in f:
				if splitnum in files:
					trainfile.write(line)
					copyfile("/home/runefeather/Desktop/Classwork/AI/Project/BreaKHis_data/" +line.split(' ')[0], "/home/runefeather/Desktop/Classwork/AI/Project/inception/data/tumor/" + splitnum +"/training/" + line.split(' ')[0].split('/')[-1])
				
	print("Training done")
	for files in val:
		with open(files) as f:
			for line in f:
				if splitnum in files:
					valfile.write(line)
					copyfile("/home/runefeather/Desktop/Classwork/AI/Project/BreaKHis_data/" +line.split(' ')[0], "/home/runefeather/Desktop/Classwork/AI/Project/inception/data/tumor/" + splitnum +"/validation/" + line.split(' ')[0].split('/')[-1])
			
	print("Validation done")

	trainfile.close()
	testfile.close()
	valfile.close()
	print("all done!")
# this function will take in the name of the main directory
def preprocessing(directory):
	testList = list()
	trainingList = list()
	validationList = list()
	path =""

	# Crawler
	for root, dirs, files in os.walk(directory):
		# print(root)
		if "non_shuffled" in root:
			pass
		else:
			for f in files:
				if "test.txt" in f:
					path = os.path.join(root, f)
					testList.append(path)
				elif "train.txt" in f:
					path = os.path.join(root, f)
					trainingList.append(path)
				elif "val.txt" in f:
					path = os.path.join(root, f)
					validationList.append(path)

	return testList, trainingList, validationList

# if __name__ == '__main__':
# 	print("STARTING NOW")
# 	test, train, val = preprocessing("/home/runefeather/Desktop/Classwork/AI/Project/breakhissplits_v2/train_val_test_60_12_28/")
# 	split(test, train, val, "split1")


