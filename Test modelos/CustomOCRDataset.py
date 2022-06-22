import os
from torchvision.io import read_image
import torch
import torch.nn.functional as F

class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, output_dir, transform=None, target_transform=None, preload_images=True, pad_to_aspect_ratio=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transform = transform
        self.target_transform = target_transform
        self.preload_images = preload_images
        self.pad_to_aspect_ratio = pad_to_aspect_ratio
        
        self.input_img = sorted(os.listdir(self.input_dir))
        self.output_label = sorted(os.listdir(self.output_dir))
        
        assert len(self.input_img) == len(self.output_label), 'Number of images does not match number of labels'
        
        self.x = []
        self.y = []
        if self.preload_images:
            for each in self.input_img:
                image = read_image(os.path.join(self.input_dir, each))
                if self.pad_to_aspect_ratio:
                    image = pad2aspect(image, aspect_ratio=self.pad_to_aspect_ratio)
                if self.transform:
                    image = self.transform(image)
                self.x.append(image)
                
            for each in self.output_label:
                label = self.load_label(os.path.join(self.output_dir, each))
                self.y.append(label)

    def __len__(self):
        return len(self.input_img)

    def __getitem__(self, idx):
        
        if self.preload_images:
            image = self.x[idx]
            label = self.y[idx]
        else:
            img_path = os.path.join(self.input_dir, self.input_img[idx])
            image = read_image(img_path)
            if self.pad_to_aspect_ratio:
                image = pad2aspect(image, aspect_ratio=self.pad_to_aspect_ratio)
            if self.transform:
                image = self.transform(image)
        
            label = self.load_label(os.path.join(self.output_dir, self.output_label[idx]))

        return image, label
        
    def load_label(self, label_path):
        label = []
        with open(label_path) as txt_file:
            for line in txt_file:
                label.append([int(line.split()[0]), *list(map(float, line.split()[1:]))])
        if self.target_transform:
            label = self.target_transform(label)
            
        return label

def pad2aspect(image, aspect_ratio=1.0):
    w, h = image.shape[-2:]
    if not (w == 0 or h == 0):
        diff_aspect_ratio = aspect_ratio - w/h
        w_target = (diff_aspect_ratio>0) * h + (diff_aspect_ratio<=0) * w
        h_target = (diff_aspect_ratio>0) * h + (diff_aspect_ratio<=0) * w

        image = F.pad(image, (int((h_target-h)/2), int((h_target-h)/2),
                              int((w_target-w)/2), int((w_target-w)/2)))

        return image

    else:
        return None