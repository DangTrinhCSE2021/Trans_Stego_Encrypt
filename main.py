# encoding: utf-8
"""
@author: yongzhi li
@contact: yongzhili@vip.qq.com

@version: 1.0
@file: main.py
@time: 2018/3/20

"""

import argparse
import os
import shutil
import socket
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import utils.transformed as transforms
from data.ImageFolderDataset import MyImageFolder
from models.HidingUNet import UnetGenerator
from models.RevealNet import RevealNet
from utils.image_encryption import recursive_permutation_encrypt, recursive_permutation_decrypt

# Change to relative path
DATA_DIR = './data'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="train",
                    help='train | val | test')
parser.add_argument('--workers', type=int, default=8,
                    help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=32,
                    help='input batch size')
parser.add_argument('--imageSize', type=int, default=256,
                    help='the number of frames')
parser.add_argument('--niter', type=int, default=100,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate, default=0.001')
parser.add_argument('--decay_round', type=int, default=10,
                    help='learning rate decay 0.5 each decay_round')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--Hnet', default='',
                    help="path to Hidingnet (to continue training)")
parser.add_argument('--Rnet', default='',
                    help="path to Revealnet (to continue training)")
parser.add_argument('--trainpics', default='./training/',
                    help='folder to output training images')
parser.add_argument('--validationpics', default='./training/',
                    help='folder to output validation images')
parser.add_argument('--testPics', default='./training/',
                    help='folder to output test images')
parser.add_argument('--outckpts', default='./training/',
                    help='folder to output checkpoints')
parser.add_argument('--outlogs', default='./training/',
                    help='folder to output images')
parser.add_argument('--outcodes', default='./training/',
                    help='folder to save the experiment codes')
parser.add_argument('--beta', type=float, default=0.75,
                    help='hyper parameter of beta')
parser.add_argument('--remark', default='', help='comment')
parser.add_argument('--test', default='', help='test mode, you need give the test pics dirs in this param')
parser.add_argument('--hostname', default=socket.gethostname().replace('*', ''), help='the host name of the running server')
parser.add_argument('--debug', type=bool, default=False, help='debug mode do not create folders')
parser.add_argument('--logFrequency', type=int, default=10, help='the frequency of print the log on the console')
parser.add_argument('--resultPicFrequency', type=int, default=100, help='the frequency of save the resultPic')
parser.add_argument('--encrypt', type=bool, default=False,
                    help='whether to encrypt images before processing')
parser.add_argument('--encrypt_rounds', type=int, default=3,
                    help='number of encryption rounds')
parser.add_argument('--encrypt_seed', type=int, default=42,
                    help='encryption seed for reproducibility')

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# print the structure and parameters number of the net
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(str(net), logPath)
    print_log('Total number of parameters: %d' % num_params, logPath)


# save code of current experiment
def save_current_codes(des_path):
    main_file_path = os.path.realpath(__file__)  # eg：/n/liyz/videosteganography/main.py
    cur_work_dir, mainfile = os.path.split(main_file_path)  # eg：/n/liyz/videosteganography/

    new_main_path = os.path.join(des_path, mainfile)
    shutil.copyfile(main_file_path, new_main_path)

    data_dir = cur_work_dir + "/data/"
    new_data_dir_path = des_path + "/data/"
    shutil.copytree(data_dir, new_data_dir_path)

    model_dir = cur_work_dir + "/models/"
    new_model_dir_path = des_path + "/models/"
    shutil.copytree(model_dir, new_model_dir_path)

    utils_dir = cur_work_dir + "/utils/"
    new_utils_dir_path = des_path + "/utils/"
    shutil.copytree(utils_dir, new_utils_dir_path)


def main():
    ############### define global parameters ###############
    global opt, optimizerH, optimizerR, writer, logPath, schedulerH, schedulerR, val_loader, smallestLoss
    global Hnet, Rnet, criterion  # Add these networks as global variables

    #################  output configuration   ###############
    opt = parser.parse_args()

    # Check CUDA availability and set device
    opt.cuda = opt.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if opt.cuda else "cpu")
    
    if not opt.cuda:
        print("CUDA is not available, using CPU instead")
    elif torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")

    cudnn.benchmark = True

    ############  create dirs to save the result #############
    if not opt.debug:
        try:
            cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
            experiment_dir = opt.hostname + "_" + cur_time + opt.remark
            opt.outckpts += experiment_dir + "/checkPoints"
            opt.trainpics += experiment_dir + "/trainPics"
            opt.validationpics += experiment_dir + "/validationPics"
            opt.outlogs += experiment_dir + "/trainingLogs"
            opt.outcodes += experiment_dir + "/codes"
            opt.testPics += experiment_dir + "/testPics"
            if not os.path.exists(opt.outckpts):
                os.makedirs(opt.outckpts)
            if not os.path.exists(opt.trainpics):
                os.makedirs(opt.trainpics)
            if not os.path.exists(opt.validationpics):
                os.makedirs(opt.validationpics)
            if not os.path.exists(opt.outlogs):
                os.makedirs(opt.outlogs)
            if not os.path.exists(opt.outcodes):
                os.makedirs(opt.outcodes)
            if (not os.path.exists(opt.testPics)) and opt.test != '':
                os.makedirs(opt.testPics)

        except OSError:
            print("mkdir failed   XXXXXXXXXXXXXXXXXXXXX")

    logPath = opt.outlogs + '/%s_%d_log.txt' % (opt.dataset, opt.batchSize)

    print_log(str(opt), logPath)
    save_current_codes(opt.outcodes)

    if opt.test == '':
        # tensorboardX writer
        writer = SummaryWriter(comment='_' + opt.remark if opt.remark else '')
        ##############   get dataset   ############################
        traindir = os.path.join(DATA_DIR, 'train')
        valdir = os.path.join(DATA_DIR, 'val')
        train_dataset = MyImageFolder(
            traindir,
            transforms.Compose([
                transforms.Resize([opt.imageSize, opt.imageSize]),
                transforms.ToTensor(),
            ]))
        val_dataset = MyImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize([opt.imageSize, opt.imageSize]),
                transforms.ToTensor(),
            ]))
        assert train_dataset
        assert val_dataset
    else:
        opt.Hnet = "./checkPoint/netH_epoch_73,sumloss=0.000447,Hloss=0.000258.pth"
        opt.Rnet = "./checkPoint/netR_epoch_73,sumloss=0.000447,Rloss=0.000252.pth"
        testdir = opt.test
        test_dataset = MyImageFolder(
            testdir,
            transforms.Compose([
                transforms.Resize([opt.imageSize, opt.imageSize]),
                transforms.ToTensor(),
            ]))
        assert test_dataset

    # Initialize models
    Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid)
    Rnet = RevealNet(output_function=nn.Sigmoid)

    # Move models to device
    if opt.cuda:
        Hnet = Hnet.cuda()
        Rnet = Rnet.cuda()

    # Initialize weights
    Hnet.apply(weights_init)
    Rnet.apply(weights_init)

    # Load pre-trained weights if specified
    if opt.Hnet != "":
        Hnet.load_state_dict(torch.load(opt.Hnet, map_location=device))
    if opt.Rnet != "":
        Rnet.load_state_dict(torch.load(opt.Rnet, map_location=device))

    # Wrap models in DataParallel if using multiple GPUs
    if opt.ngpu > 1 and opt.cuda:
        Hnet = torch.nn.DataParallel(Hnet)
        Rnet = torch.nn.DataParallel(Rnet)

    print_network(Hnet)
    print_network(Rnet)

    # MSE loss
    criterion = nn.MSELoss()
    if opt.cuda:
        criterion = criterion.cuda()

    # training mode
    if opt.test == '':
        # setup optimizer
        optimizerH = optim.Adam(Hnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        optimizerR = optim.Adam(Rnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        schedulerH = ReduceLROnPlateau(optimizerH, mode='min', factor=0.2, patience=5, verbose=True)
        schedulerR = ReduceLROnPlateau(optimizerR, mode='min', factor=0.2, patience=5, verbose=True)

        train_loader = DataLoader(train_dataset, batch_size=opt.batchSize,
                                shuffle=True, num_workers=int(opt.workers))
        val_loader = DataLoader(val_dataset, batch_size=opt.batchSize,
                              shuffle=True, num_workers=int(opt.workers))

        # training
        print("training is beginning .....................................................")
        smallestLoss = 10000
        try:
            for epoch in range(opt.niter):
                train(train_loader, epoch)
                val_hloss, val_rloss, val_sumloss = validation(val_loader, epoch)
                schedulerH.step(val_hloss)
                schedulerR.step(val_rloss)

                # save the best model
                if val_sumloss < smallestLoss:
                    smallestLoss = val_sumloss
                    torch.save(Hnet.state_dict(),
                             '%s/netH_epoch_%d,sumloss=%.6f,Hloss=%.6f.pth' % (
                                 opt.outckpts, epoch, val_sumloss, val_hloss))
                    torch.save(Rnet.state_dict(),
                             '%s/netR_epoch_%d,sumloss=%.6f,Rloss=%.6f.pth' % (
                                 opt.outckpts, epoch, val_sumloss, val_rloss))
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise  # Re-raise the exception to see the full traceback

    # test mode
    else:
        test_loader = DataLoader(test_dataset, batch_size=opt.batchSize,
                               shuffle=False, num_workers=int(opt.workers))
        test(test_loader, 0)
        print("##################   test is completed, the result pic is saved in the ./training/yourcompuer+time/testPics/   ######################")


def train(train_loader, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Hlosses = AverageMeter()  # hiding loss
    Rlosses = AverageMeter()  # reveal loss
    SumLosses = AverageMeter()  # sum of hiding loss and reveal loss

    # switch to train mode
    Hnet.train()
    Rnet.train()

    end = time.time()
    for i, (cover_img, secret_img) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        batch_size = cover_img.size(0)

        if opt.cuda:
            cover_img = cover_img.cuda()
            secret_img = secret_img.cuda()

        # Encrypt images if encryption is enabled
        if opt.encrypt:
            try:
                secret_img_encrypted = recursive_permutation_encrypt(
                    secret_img, opt.encrypt_rounds, opt.encrypt_seed, batch_idx=i
                )
            except Exception as e:
                print(f"Encryption error: {str(e)}")
                secret_img_encrypted = secret_img
        else:
            secret_img_encrypted = secret_img

        # concatenate
        concat_img = torch.cat([cover_img, secret_img_encrypted], dim=1)

        # hiding net
        Hout = Hnet(concat_img)

        # reveal net
        Rout = Rnet(Hout)

        # Decrypt output if encryption was used
        if opt.encrypt:
            try:
                Rout_decrypted = recursive_permutation_decrypt(
                    Rout, opt.encrypt_rounds, opt.encrypt_seed, batch_idx=i
                )
            except Exception as e:
                print(f"Decryption error: {str(e)}")
                Rout_decrypted = Rout
        else:
            Rout_decrypted = Rout

        # loss
        Hloss = criterion(Hout, cover_img)
        Rloss = criterion(Rout_decrypted, secret_img)

        # sum of the two losses
        sum_loss = opt.beta * Hloss + (1 - opt.beta) * Rloss

        # compute gradient and do optim step
        optimizerH.zero_grad()
        optimizerR.zero_grad()
        sum_loss.backward()
        optimizerH.step()
        optimizerR.step()

        # record loss
        Hlosses.update(Hloss.item(), batch_size)
        Rlosses.update(Rloss.item(), batch_size)
        SumLosses.update(sum_loss.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.logFrequency == 0:
            print_log('Train: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Sum_Loss {SumLosses.val:.4f} ({SumLosses.avg:.4f})\t'
                  'H_Loss {Hlosses.val:.4f} ({Hlosses.avg:.4f})\t'
                  'R_Loss {Rlosses.val:.4f} ({Rlosses.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time,
                SumLosses=SumLosses, Hlosses=Hlosses, Rlosses=Rlosses), logPath)

        if i % opt.resultPicFrequency == 0:
            save_result_pic(opt.trainpics, cover_img, secret_img, Hout, Rout_decrypted, epoch, i, True)

    # log values for each epoch
    writer.add_scalar('Train/Sum_Loss', SumLosses.avg, epoch)
    writer.add_scalar('Train/H_Loss', Hlosses.avg, epoch)
    writer.add_scalar('Train/R_Loss', Rlosses.avg, epoch)

    return Hlosses.avg, Rlosses.avg, SumLosses.avg


def validation(val_loader, epoch):
    print(
        "#################################################### validation begin ########################################################")
    start_time = time.time()
    Hnet.eval()
    Rnet.eval()
    Hlosses = AverageMeter()
    Rlosses = AverageMeter()
    
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            all_pics = data
            this_batch_size = int(all_pics.size()[0] / 2)

            cover_img = all_pics[0:this_batch_size, :, :, :]
            secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :]

            # Move to CUDA first if enabled
            if opt.cuda:
                cover_img = cover_img.cuda()
                secret_img = secret_img.cuda()

            # Encrypt images if enabled (after CUDA transfer)
            if opt.encrypt:
                try:
                    cover_img = recursive_permutation_encrypt(cover_img, opt.encrypt_rounds, opt.encrypt_seed)
                    secret_img = recursive_permutation_encrypt(secret_img, opt.encrypt_rounds, opt.encrypt_seed)
                except Exception as e:
                    print_log(f"Validation encryption error: {str(e)}", logPath)
                    continue

            # Concatenate images after encryption
            concat_img = torch.cat([cover_img, secret_img], dim=1)
            if opt.cuda:
                concat_img = concat_img.cuda()

            # Forward pass
            container_img = Hnet(concat_img)
            errH = criterion(container_img, cover_img)
            Hlosses.update(errH.item(), this_batch_size)

            rev_secret_img = Rnet(container_img)

            # Decrypt revealed secret image if encryption is enabled
            if opt.encrypt:
                try:
                    rev_secret_img = recursive_permutation_decrypt(rev_secret_img, opt.encrypt_rounds, opt.encrypt_seed)
                    secret_img = recursive_permutation_decrypt(secret_img, opt.encrypt_rounds, opt.encrypt_seed)
                except Exception as e:
                    print_log(f"Validation decryption error: {str(e)}", logPath)
                    continue

            errR = criterion(rev_secret_img, secret_img)
            Rlosses.update(errR.item(), this_batch_size)

            if i % 50 == 0:
                save_result_pic(this_batch_size, cover_img, container_img, secret_img, rev_secret_img, epoch, i,
                                opt.validationpics)

    val_hloss = Hlosses.avg
    val_rloss = Rlosses.avg
    val_sumloss = val_hloss + opt.beta * val_rloss

    val_time = time.time() - start_time
    val_log = "validation[%d] val_Hloss = %.6f\t val_Rloss = %.6f\t val_Sumloss = %.6f\t validation time=%.2f" % (
        epoch, val_hloss, val_rloss, val_sumloss, val_time)
    print_log(val_log, logPath)

    if not opt.debug:
        writer.add_scalar('validation/H_loss_avg', Hlosses.avg, epoch)
        writer.add_scalar('validation/R_loss_avg', Rlosses.avg, epoch)
        writer.add_scalar('validation/sum_loss_avg', val_sumloss, epoch)

    print(
        "#################################################### validation end ########################################################")
    return val_hloss, val_rloss, val_sumloss


def test(test_loader, epoch):
    print(
        "#################################################### test begin ########################################################")
    start_time = time.time()
    Hnet.eval()
    Rnet.eval()
    Hlosses = AverageMeter()
    Rlosses = AverageMeter()
    
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            all_pics = data
            this_batch_size = int(all_pics.size()[0] / 2)

            cover_img = all_pics[0:this_batch_size, :, :, :]
            secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :]

            # Move to CUDA first if enabled
            if opt.cuda:
                cover_img = cover_img.cuda()
                secret_img = secret_img.cuda()

            # Encrypt images if enabled (after CUDA transfer)
            if opt.encrypt:
                try:
                    cover_img = recursive_permutation_encrypt(cover_img, opt.encrypt_rounds, opt.encrypt_seed)
                    secret_img = recursive_permutation_encrypt(secret_img, opt.encrypt_rounds, opt.encrypt_seed)
                except Exception as e:
                    print_log(f"Test encryption error: {str(e)}", logPath)
                    continue

            # Concatenate images after encryption
            concat_img = torch.cat([cover_img, secret_img], dim=1)
            if opt.cuda:
                concat_img = concat_img.cuda()

            # Forward pass
            container_img = Hnet(concat_img)
            errH = criterion(container_img, cover_img)
            Hlosses.update(errH.item(), this_batch_size)

            rev_secret_img = Rnet(container_img)

            # Decrypt revealed secret image if encryption is enabled
            if opt.encrypt:
                try:
                    rev_secret_img = recursive_permutation_decrypt(rev_secret_img, opt.encrypt_rounds, opt.encrypt_seed)
                    secret_img = recursive_permutation_decrypt(secret_img, opt.encrypt_rounds, opt.encrypt_seed)
                except Exception as e:
                    print_log(f"Test decryption error: {str(e)}", logPath)
                    continue

            errR = criterion(rev_secret_img, secret_img)
            Rlosses.update(errR.item(), this_batch_size)

            save_result_pic(this_batch_size, cover_img, container_img, secret_img, rev_secret_img, epoch, i,
                            opt.testPics)

    val_hloss = Hlosses.avg
    val_rloss = Rlosses.avg
    val_sumloss = val_hloss + opt.beta * val_rloss

    val_time = time.time() - start_time
    val_log = "test[%d] test_Hloss = %.6f\t test_Rloss = %.6f\t test_Sumloss = %.6f\t test time=%.2f" % (
        epoch, val_hloss, val_rloss, val_sumloss, val_time)
    print_log(val_log, logPath)

    print(
        "#################################################### test end ########################################################")
    return val_hloss, val_rloss, val_sumloss


# print training log and save into logFiles
def print_log(log_info, log_path, console=True):
    if console:
        print(log_info)
    if not opt.debug:
        with open(log_path, 'a+') as f:
            f.writelines(log_info + '\n')


# save result pics, coverImg filePath and secretImg filePath
def save_result_pic(save_path, cover_img, secret_img, container_img, rev_secret_img, epoch, i, is_train=True):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Ensure tensors are on CPU and detached from computation graph
    cover_img = cover_img.cpu().detach()
    secret_img = secret_img.cpu().detach()
    container_img = container_img.cpu().detach()
    rev_secret_img = rev_secret_img.cpu().detach()
    
    # Take only the first 8 images if batch size is larger
    if cover_img.size(0) > 8:
        cover_img = cover_img[:8]
        secret_img = secret_img[:8]
        container_img = container_img[:8]
        rev_secret_img = rev_secret_img[:8]
    
    # Concatenate all images horizontally
    pic = torch.cat([cover_img, container_img, secret_img, rev_secret_img], dim=0)
    
    if is_train:
        vutils.save_image(pic, '%s/epoch_%03d_iter_%03d.png' % (save_path, epoch, i), normalize=True)
    else:
        vutils.save_image(pic, '%s/iter_%03d.png' % (save_path, i), normalize=True)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
