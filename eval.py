import os
import torch
from config import Config
from load_data import loadtestdata
from torch.autograd import Variable
from DMUnet import DMNestUnet
import matplotlib.pyplot as plt
import PIL.Image as pil_image
import numpy as np
from utils import rgb2RGGB, calc_psnr, Crop, unCrop, imgSIZEnormalize

CONFIG = Config()
device = CONFIG.DEVICE


def test_for_datasets(n, param_root, data_path, oup_path):

    import imageio
    import shutil
    import skimage.io as skio
    Model = torch.load(os.path.join(param_root,'params/model_e{}.pkl'.format(n)), map_location=device)
    net = Model.to(device)
    L = CONFIG.UnetL
    if not os.path.exists(oup_path):
        os.mkdir(oup_path)
    for i in range(L):
        os.mkdir(os.path.join(oup_path, 'L'+str(i+1)))
    for root, dirs, files in os.walk(data_path):
        if dirs:
            for dir in dirs:
                for i in range(L):
                    os.mkdir(os.path.join(oup_path, 'L'+str(i+1), dir))
                class_rd_path = os.path.join(data_path, dir)
                for _, __, images in os.walk(os.path.join(data_path, dir)):
                    for image in images:
                        img_name = image.split('.')[0]
                        img_path = os.path.join(class_rd_path, image)
                        try:
                            gt = (np.array(imageio.imread(img_path),dtype=np.float32) / 255.0).transpose(2, 0, 1)
                            gt = torch.from_numpy(gt).unsqueeze(0)
                            gt = imgSIZEnormalize(gt)
                            input = rgb2RGGB(gt).to(device)
                            with torch.no_grad():
                                output = net(Variable(input))
                                for i, img in enumerate(output):
                                    out_pic = img.squeeze(0).cpu().numpy().clip(0, 1).transpose(1, 2, 0)
                                    skio.imsave(os.path.join(oup_path, 'L'+str(i+1), dir, img_name + '.bmp'), out_pic)
                        except:
                            print('   Error with image:{}.'.format(img_path))
                            for i in range(L):
                                shutil.copy(img_path, os.path.join(oup_path, 'L'+str(i+1), dir))
                    print('Finish file {}.'.format(dir))
        else:
            for image in files:
                img_name = image.split('.')[0]
                img_path = os.path.join(data_path, image)
                try:
                    gt = (np.array(imageio.imread(img_path), dtype=np.float32) / 255.0).transpose(2, 0, 1)
                    gt = torch.from_numpy(gt).unsqueeze(0)
                    gt = imgSIZEnormalize(gt)
                    input = rgb2RGGB(gt).to(device)
                    with torch.no_grad():
                        output = net(Variable(input))
                        for i, img in enumerate(output):
                            out_pic = img.squeeze(0).cpu().numpy().clip(0, 1).transpose(1, 2, 0)
                            skio.imsave(os.path.join(oup_path, 'L'+str(i+1), img_name + '.bmp'), out_pic)
                except:
                    print('   Error with image:{}.'.format(img_path))
                    for i in range(L):
                        shutil.copy(img_path, os.path.join(oup_path, 'L' + str(i + 1)))


def test(testloader, net, single=False, crop=0, ignore=0):

    L = CONFIG.DMUnetL
    oup_path = os.path.join(CONFIG.OUTPIC_FILE, CONFIG.Test_DataSet)
    if not os.path.exists(oup_path):
        os.mkdir(oup_path)

    psnr = [0 for i in range(L)]
    pic_psnr = [0 for i in range(L)]
    CPSNR = np.zeros((L, 4))
    # psnr = 0
    if single:
        from openpyxl import Workbook
        wkbook = Workbook()
        bksheet = []
        for j in range(L):
            bksheet.append(wkbook.create_sheet('L'+str(j+1), j))
    with torch.no_grad():
        for i, data in enumerate(testloader,0):
            images, labels = data
            images = imgSIZEnormalize(images)
            ground_truth = images
            if crop > 0:
                images = Crop(images)
                inputs = rgb2RGGB(images).to(device)
                outputs = net(Variable(inputs))
                outputs = unCrop(outputs)
            else:
                inputs = rgb2RGGB(images).to(device)
                outputs = net(Variable(inputs))

            ground_truth = ground_truth.to(device)
            for j in range(L):
                output = outputs[j]
                pic_psnr[j] = calc_psnr(output, ground_truth, ignore=ignore)
                psnr[j] += pic_psnr[j]

                if single:
                    temp = calc_psnr(output, ground_truth, ignore=ignore, cpsnr=True)
                    bksheet[j].append(['PIC {}'.format(i+1)] + list(temp))
                    CPSNR[j, :] += temp
                    output = (output.clamp(0.0, 1.0)*255).cpu().squeeze(0)
                    output = np.array(output, dtype=np.uint8).transpose([1, 2, 0])
                    output = pil_image.fromarray(output)
                    if not os.path.exists(os.path.join(oup_path, 'L{}'.format(j+1))):
                        os.mkdir(os.path.join(oup_path, 'L{}'.format(j+1)))
                    output.save(os.path.join(oup_path, 'L{}'.format(j+1), '{}.bmp'.format(i+1)))
            if single:
                print('No.{} pic... psnr = {} dB.'.format(i+1, np.array(pic_psnr)))
        for j in range(L):
            psnr[j] /= (i+1)
        if single:
            CPSNR /= (i+1)
            for j in range(L):
                bksheet[j].append(['Ave.'] + list(CPSNR[j, :]))
            wkbook_path = os.path.join(oup_path, CONFIG.Test_DataSet + '_cpsnr_crop_{}_ignore_{}.xlsx'.format(crop, ignore))
            if os.path.exists(wkbook_path):
                os.remove(wkbook_path)
            wkbook.save(wkbook_path)
            print(CPSNR)
            print(psnr)
            print('All {} pics'.format(i+1))
        return np.array(psnr)


def test_datasets(model_root, data_root, crop=0, ignore=0, device=CONFIG.DEVICE):

    test_data = loadtestdata(data_root)
    Model = DMNestUnet(in_channels=CONFIG.INP, n_classes=CONFIG.OUP)
    Model.load_state_dict(torch.load(model_root, map_location=device))
    Model = Model.to(device)
    __ = test(test_data, Model, single=True, crop=crop, ignore=ignore)



def test_for_time(model_root, data_root, device=CONFIG.DEVICE):

    import time

    test_file = data_root
    testloader = loadtestdata(test_file)
    Model = DMNestUnet(in_channels=CONFIG.INP, n_classes=CONFIG.OUP)
    Model.load_state_dict(torch.load(model_root, map_location=device))  # load pre-trained .pkl model
    Model = Model.to(device)
    test_time = 0.0

    with torch.no_grad():

        for i, data in enumerate(testloader, 0):

            images, labels = data

            if 'cuda' in device.__str__():
                torch.cuda.synchronize()

            start = time.time()
            inputs = rgb2RGGB(images).to(device)
            outputs = Model(Variable(inputs))

            if 'cuda' in device.__str__():
                torch.cuda.synchronize()

            fin = time.time()
            deta_time = fin - start

            print('Finish No.{} pic... TIME: {}s.'.format(i+1, deta_time))
            test_time += deta_time

    print('Finish ALL pic... TIME: {}s.'.format(test_time))



if __name__ == "__main__":

    model_path = CONFIG.TEST_PARAM_ROOT
    data_path = os.path.join(CONFIG.TEST_DATA_ROOT, CONFIG.Test_DataSet)
    test_datasets(model_path, data_path, ignore=5)
    # test_for_time(model_path, data_path)

