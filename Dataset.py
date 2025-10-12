

import torch
from torch.utils import data
from torchvision import transforms as T
from torch.utils.data import DataLoader
from config import opt
from skimage import io, color
import random

class Train_Data(data.Dataset):
    def __init__(self):
        self.data_root = opt.data_root
        self.HR_root = opt.HR_root
        self.label_root=opt.label_root

    def __getitem__(self, index):
        img_index = random.randint(1, 3064)

        img_H = io.imread(self.HR_root + str(img_index) + '.png')
        img_L = io.imread(self.data_root + str(img_index) + '.png')
        label = io.imread(self.label_root + str(img_index) + '.png')

        img_H_y = img_H / 255
        img_L_y = img_L / 255

        HR = torch.FloatTensor(img_H_y).unsqueeze(0)
        LR_image = torch.FloatTensor(img_L_y).unsqueeze(0)
        label_image = torch.FloatTensor(label).unsqueeze(0)

        return LR_image, HR, label_image

    def __len__(self):
        return opt.num_data


if __name__ == '__main__':
    train_data = Train_Data()

    train_loader = DataLoader(train_data, 8)

    for i, (data, HR,label) in enumerate(train_loader):
        print(i)
        if i == 4:
            data_img = T.ToPILImage()(data)
            data_img.show()
            HR_img = T.ToPILImage()(HR)
            HR_img.show()
            label_img=T.ToPILImage()(label)
            label_img.show()
            print(data)
            print(HR)
            print(label)
            print(data.size())
            print(HR.size())
            print(label.size())
            break













