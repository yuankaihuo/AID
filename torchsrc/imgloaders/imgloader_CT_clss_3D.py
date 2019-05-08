import os
import numpy as np
from torch.utils import data
import nibabel as nib
from image_manipulation import *
from glob import glob


# output_x = 168
# output_y = 168
# output_z = 64



class pytorch_loader_clss3D(data.Dataset):
    def __init__(self, subdict, num_labels,input_root_dir,res,imsize,dual_network=False,data_augmentation=False):
        self.subdict = subdict
        self.img_subs = subdict['img_subs']
        self.img_files = subdict['img_files']
        self.input_root_dir = input_root_dir
        self.categories = subdict['categories']
        self.num_labels = num_labels
        self.output_x = res[0]
        self.output_y = res[1]
        self.output_z = res[2]
        self.img_x = imsize[0]
        self.img_y = imsize[1]
        self.img_z = imsize[2]
        self.data_augmentation = data_augmentation
        self.dual_network = dual_network

    def __getitem__(self, index):
        num_labels = self.num_labels
        input_root_dir = self.input_root_dir

        labels = range(num_labels)
        sub_name = self.img_subs[index]
        img_file = self.img_files[index]
        img_file_name = os.path.basename(img_file)
        img_file_normalized = os.path.join(input_root_dir,img_file_name)
        img_3d = nib.load(img_file_normalized)
        try:
            img = img_3d.get_data()
        except:
            print('******************** %s\n'%img_file_normalized)

        img = np.transpose(img, (2, 0, 1))
        img = (img - img.min()) / (img.max() - img.min())

        if self.data_augmentation:
            rand = random.uniform(0, 1)
            if (0 <= rand < 0.5):
                img = random_rotation(img)
                img = random_translation(img, 5)

        img = img*255.0

        if self.dual_network:
            fname_strs = img_file_name.split('-x-')
            img2_file_normalized = glob(os.path.join(input_root_dir, fname_strs[0]+'*CAC2.nii.gz'))
            img2_3d = nib.load(img2_file_normalized[0])
            img2 = img2_3d.get_data()
            img2 = np.transpose(img2, (2, 0, 1))
            img2 = (img2 - img2.min()) / (img2.max() - img2.min())

            if self.data_augmentation:
                rand = random.uniform(0, 1)
                if (0 <= rand < 0.5):
                    img2 = random_rotation(img2)
                    img2 = random_translation(img2, 2)

            img2 = img2 * 255.0

            x = np.zeros((2, self.img_z, self.img_x, self.img_y))
            x[0, 0:self.output_z, 0:self.output_x, 0:self.output_y] = img[0:self.output_z, 0:self.output_x,
                                                                      0:self.output_y]
            x[1, 0:self.output_z, 0:self.output_x, 0:self.output_y] = img2[0:self.output_z, 0:self.output_x,
                                                                      0:self.output_y]
        else:
            x = np.zeros((1, self.img_z, self.img_x, self.img_y))
            x[0,0:self.output_z,0:self.output_x,0:self.output_y] = img[0:self.output_z,0:self.output_x,0:self.output_y]
        x = x.astype('float32')
        y = self.categories[index]

        return x, y, sub_name

    def __len__(self):
        return len(self.img_subs)