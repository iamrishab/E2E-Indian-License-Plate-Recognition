import os
import cv2
import re
from pdb import set_trace

path_to_dataset = 'endtoend'
path_to_save_folder = 'all'
supported_img_formats = ['.jpg', '.jpeg', '.png']

os.makedirs(path_to_save_folder, exist_ok=True)

for root, folders, files in os.walk(path_to_dataset):
	for folder in folders:
		if not folder.startswith('.'):
			for file in os.listdir(os.path.join(root, folder)):
				name, ext = os.path.splitext(file)
				if ext.lower() in supported_img_formats:
					print(f'Processing image: {os.path.join(root, folder, file)}')
					img = cv2.imread(os.path.join(root, folder, file))
					with open(os.path.join(root, folder, f"{name}.txt")) as fr:
						# set_trace()
						line = fr.readline()
						label = line.split('\t')[-1].strip()
						x, y, w, h = map(int, line.split('\t')[1:-1])
						save_path = os.path.join(path_to_save_folder, f'{label}.jpg')
						cv2.imwrite(save_path, img[y:y+h, x:x+w, :])
						crop = cv2.imread(save_path)
						try:
							height, width, _ = crop.shape
							print('image checked:', file)
						except Exception as e:
							os.remove(save_path)
							print('image check failed:', file)
