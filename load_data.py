import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from config import Config
import cv2

config = Config()


def loadtraindata(train_path):
    preprocess = transforms.Compose([#transforms.CenterCrop(config.INPUT_SIZE),
                                     # transforms.Resize(config.INPUT_SIZE),
                                     # transforms.RandomHorizontalFlip(p=0.3),
                                     # transforms.RandomRotation(10),
                                     transforms.ToTensor(),
                                     # transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
                                     ])
    trainset = torchvision.datasets.ImageFolder(train_path,
                                                 preprocess)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.TRAIN_BATCH,
                                              shuffle=True, num_workers=config.Workers)
    return trainloader
#

def loadtestdata(test_path):
    preprocess = transforms.Compose([# ResizeCrop(config.INPUT_SIZE),
                                     transforms.ToTensor(),
                                     # transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
                                     # transforms.Normalize(mean=[0.5], std=[0.5])
                                     ])
    testset = torchvision.datasets.ImageFolder(test_path, preprocess)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.TEST_BATCH,
                                              shuffle=False, num_workers=config.Workers)
    return testloader


class ResizeCrop:
    def __init__(self, img_size=config.INPUT_SIZE):
        self.size = img_size

    def __call__(self, input_im):
        input_img = np.array(input_im)
        h, w, c = input_img.shape
        # h = float(h)
        # w = float(w)
        if h/w == self.size[0]/self.size[1]:#important!!!
            img_crop = cv2.resize(input_img, (self.size[1],self.size[0]))
        else:
            fac = np.array([self.size[0]/h, self.size[1]/w],dtype='float32')
            scale , idx = torch.max(torch.from_numpy(fac), 0)
            scale = scale.numpy()
            if idx == 1:
                input_img = cv2.resize(input_img, (self.size[1], int(np.ceil(scale*h))))
                start_pos = int(np.floor(int(np.floor(scale*h))/2 - self.size[0]/2))
                img_crop = input_img[start_pos : start_pos+self.size[0], :, :]
            else:
                input_img = cv2.resize(input_img, (int(np.ceil(scale*w)), self.size[0]))
                start_pos = int(np.floor(int(np.floor(scale*w))/2 - self.size[1]/2))
                img_crop = input_img[:, start_pos : start_pos+self.size[1], :]

        return img_crop



if __name__ == '__main__':
    pass

