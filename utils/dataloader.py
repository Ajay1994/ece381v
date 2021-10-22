import torch
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import pandas as pd
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True
import sys
random.seed(0)

class LesionDataLoader(Dataset):
    def __init__(self, path_to_images, mode, transform):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        
        self.label_df = pd.read_csv(path_to_images + "split_label.csv")
        self.label_df = self.label_df[self.label_df["mode"] == mode]
        
        self.img_dir = "ISIC_2019_Training_Input/"
        
        self.transform = transform
        self.path_to_images = path_to_images

    def __getitem__(self, idx):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        
        image = Image.open(self.path_to_images + self.img_dir + self.label_df.iloc[idx, 1] + ".jpg")
        image = image.convert('RGB')
        image = self.transform(image)
        
        label_index = ["MEL","NV","BCC","AK","BKL","DF","VASC","SCC","UNK"]
        label = []
        
        for i in label_index:
            label.append(self.label_df.iloc[idx, self.label_df.columns.get_loc(i)])
            
        return image, torch.LongTensor(label)

    def __len__(self):
        return len(self.label_df)
    