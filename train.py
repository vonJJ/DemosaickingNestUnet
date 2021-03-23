import torch
from torch.autograd import Variable
import os
from config import Config
import numpy as np
from DMUnet import DMNestUnet
from utils import rgb2RGGB
from learning_rate_schedulers import StepDecay
from DMUnet import DMNestUnet

config = Config()
device = config.DEVICE

def trainandsave(trainloader, net, startE=0):

    if not os.path.exists(os.path.join(config.TRAIN_PARAM_ROOT,'loss')):
        os.mkdir(os.path.join(config.TRAIN_PARAM_ROOT,'loss'))

    if not os.path.exists(os.path.join(config.TRAIN_PARAM_ROOT,'params')):
        os.mkdir(os.path.join(config.TRAIN_PARAM_ROOT,'params'))

    optimizer = config.OPTIM(net.parameters(), lr=config.LEARNING_RATE)
    criterion = config.CRITERION1
    net = net.to(device)
    LOSS = []
    flag = -1
    if startE >= config.CHANGE_CRI_EPOCH[0]:
        criterion = config.CRITERION2
        flag = 1
    for epoch in range(config.NUM_EPOCH):
        if epoch + startE == config.CHANGE_CRI_EPOCH[0]:
            criterion = config.CRITERION2
            flag = 1
        RUNNING_LOSS = 0.0
        Loss = []
        StepDecay(optimizer, epoch)
        optimizer.zero_grad()
        c = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # inputs = imgSIZEnormalize(inputs)
            inputs_bayer = rgb2RGGB(inputs)
            inputs_bayer = Variable(inputs_bayer.to(device))
            ground_truth = inputs.to(device)
            outputs = net(inputs_bayer)
            loss = 0
            for output in outputs:
                loss += criterion(output, ground_truth)
            loss /= config.DMUnetL
            loss *= flag
            running_loss = loss.item()
            RUNNING_LOSS += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            c += 1
            if i % config.ENU_NUM == config.ENU_NUM-1:
                Loss.append(running_loss)
                print('Epoch '+str(epoch + startE)+' : '+str(i//config.ENU_NUM)+' , LOSS = '+str(running_loss))
        Loss0 = np.array(Loss)
        LOSS.append(RUNNING_LOSS/c)
        np.save(config.TRAIN_PARAM_ROOT+'/loss/epoch_{}.npy'.format(epoch + startE), Loss0)
        torch.save(net, config.TRAIN_PARAM_ROOT+'/params/model_e{}.pkl'.format(epoch + startE))
        torch.save(net.state_dict(), config.TRAIN_PARAM_ROOT+'/params/params_e{}.pth'.format(epoch + startE))
    np.save(config.TRAIN_PARAM_ROOT+'/loss/whole_train.npy', np.array(LOSS))
    print('Finishked Training')
    torch.save(net, config.TRAIN_PARAM_ROOT+'/params/model.pkl')
    torch.save(net.state_dict(), config.TRAIN_PARAM_ROOT+'/params/params.pth')

if __name__ == '__main__':
    device = config.DEVICE
    E = -1  # If E = -1, trainning starts from the beginning; If E = n, then trainning continues from 'params_e{n}.pth'
    from load_data import loadtraindata
    train_path = os.path.join(Config.TRAIN_DATA_ROOT, Config.Train_DataSet)
    data_true = loadtraindata(train_path)
    net = DMNestUnet(in_channels=config.INP, n_classes=config.OUP)
    if E != -1:
        net.load_state_dict(torch.load(os.path.join(Config.TRAIN_PARAM_ROOT,'params/params_e{}.pth'.format(E)), map_location=device))
    trainandsave(data_true, net, E+1)
    net = net.cpu()