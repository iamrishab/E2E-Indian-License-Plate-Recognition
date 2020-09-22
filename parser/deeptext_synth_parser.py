import os
import glob
import shutil


synth_data_dir = '/home/rishab/licenseplaterecognition/utils/synth'
save_dir = '/home/rishab/lpr_dataset/labelled/deeptext_synth'

SUPPORTED_IMG_FORMAT = ['.png', '.jpg', '.jpeg']

CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
         'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
         'U', 'V', 'W', 'X', 'Y', 'Z'
         ]

train_split = 0.7 # train split
validation_split = 0.15 # validation split
test_split = 0.15 # test split

total_img_list = glob.glob(f"{synth_data_dir}/*.png")
num_samples = len(total_img_list)

num_img_train = int(num_samples * train_split)
num_img_val = int(num_samples * validation_split)
num_img_test = int(num_samples * test_split)

train_list = total_img_list[:num_img_train]
val_list = total_img_list[num_img_train:num_img_train+num_img_val]
test_list = total_img_list[num_img_train+num_img_val:]


def normalize_text(txt):
    # replace special characters
    for x in list(txt):
        if x not in CHARS:
            txt = txt.replace(x, '') 
    return txt

# set_trace()
save_train_dir = os.path.join(save_dir, 'train') 
os.makedirs(f'{save_train_dir}/data', exist_ok=True)
with open(os.path.join(save_train_dir, 'gt.txt'), 'w') as fw:
    for img_path in train_list:
        if os.path.exists(img_path):
            img_prefix, ext = os.path.splitext(os.path.basename(img_path))
            label = normalize_text(img_prefix)
            new_img_path = os.path.join(save_train_dir, 'data', f"{os.path.basename(label)}{ext}")
            shutil.copy(img_path, new_img_path)
            fw.write(f"{os.path.abspath(new_img_path)}\t{label}\n")

save_val_dir = os.path.join(save_dir, 'val')
os.makedirs(f'{save_val_dir}/data', exist_ok=True)
with open(os.path.join(save_val_dir, 'gt.txt'), 'w') as fw:
    for img_path in val_list:
        if os.path.exists(img_path):
            img_prefix, ext = os.path.splitext(os.path.basename(img_path))
            label = normalize_text(img_prefix)
            new_img_path = os.path.join(save_train_dir, 'data', f"{os.path.basename(label)}{ext}")
            shutil.copy(img_path, new_img_path)
            fw.write(f"{os.path.abspath(new_img_path)}\t{label}\n")

save_test_dir = os.path.join(save_dir, 'test')
os.makedirs(f'{save_test_dir}/data', exist_ok=True)
with open(os.path.join(save_test_dir, 'gt.txt'), 'w') as fw:
    for img_path in test_list:
        if os.path.exists(img_path):
            img_prefix, ext = os.path.splitext(os.path.basename(img_path))
            label = normalize_text(img_prefix)
            new_img_path = os.path.join(save_train_dir, 'data', f"{os.path.basename(label)}{ext}")
            shutil.copy(img_path, new_img_path)
            fw.write(f"{os.path.abspath(new_img_path)}\t{label}\n")

