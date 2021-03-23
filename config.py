import os
import torch.nn as nn
import torch.optim as optim
import pytorch_ssim as py_ssim
import torch

class Config:

    DEVICE = torch.device('cuda')
    # device = torch.device('cpu')

    DeepSupervision = True
    GaussSmoothingKernel = 5
    GaussFilterSigma = 5
    DMUnetL = 3
    INP = 4
    OUP = 3
    NUM_EPOCH = 300
    WindowSize = 11
    CRITERION1 = py_ssim.SSIM(window_size=WindowSize)
    CRITERION2 = nn.MSELoss()
    CHANGE_CRI_EPOCH = 10
    INPUT_SIZE = (128, 128)
    LEARNING_RATE = 0.0001
    OPTIM = optim.Adam
    FACTOR = 0.1
    DROP_EVERY = 10
    TRAIN_BATCH = 16
    TEST_BATCH = 1
    BETAS = (0.9, 0.999)
    Workers = 4 # 16

    DataSets_Name = ['McMaster', 'Kodak24', 'Urban100', 'Manga109', 'ImageNet', '128x128_pic']

    TRAIN_DATA_ROOT = './DataSets/Training'
    Train_DataSet = DataSets_Name[5]

    TEST_DATA_ROOT = './DataSets/Evaluating'
    Test_DataSet =  DataSets_Name[0]

    TRAIN_PARAM_ROOT = './TrainingResult'
    TEST_PARAM_ROOT = './pretrained_model/params_e88_convert.pth'

    OUTPIC_FILE = './TestingResult'

    ENU_NUM = 400
