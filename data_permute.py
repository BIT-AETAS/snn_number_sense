import numpy as np
import os
from PIL import Image
import argparse
np.random.seed(2022)

parser = argparse.ArgumentParser(description='Permute image data for dataset')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to the original dataset')
parser.add_argument('--save_path', type=str, required=True, help='Path to save the permuted dataset')
args = parser.parse_args()

if __name__ == '__main__':
    dataset_path = args.dataset_path
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    class_list = os.listdir(dataset_path)
    for _class in class_list:
        class_path = os.path.join(dataset_path, _class)
        save_class_path = os.path.join(save_path, _class)
        if not os.path.exists(save_class_path):
            os.makedirs(save_class_path)
        img_list = os.listdir(class_path)
        for img_name in img_list:
            if img_name.endswith('.jpg') or img_name.endswith('.png'):
                img_path = os.path.join(class_path, img_name)
                img = Image.open(img_path).convert('L')
                img_array = np.array(img)
                h, w = img_array.shape
                flat_img = img_array.flatten()
                np.random.shuffle(flat_img)
                permuted_img_array = flat_img.reshape((h, w))
                permuted_img = Image.fromarray(permuted_img_array.astype(np.uint8))
                permuted_img.save(os.path.join(save_class_path, img_name))
        # Copy info.txt
        info_src = os.path.join(class_path, 'info.txt')
        info_dst = os.path.join(save_class_path, 'info.txt')
        if os.path.exists(info_src):
            with open(info_src, 'r', encoding='utf-8') as f_src:
                with open(info_dst, 'w', encoding='utf-8') as f_dst:
                    f_dst.write(f_src.read())