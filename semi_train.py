# In[1]:
import numpy as np
import glob
import h5py
import shutil
import random

from tqdm import tqdm
from itertools import cycle
from os import listdir, rename
from os.path import isfile, join
from collections import Counter


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as tud
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

SEED = 1234
torch.manual_seed(SEED) # cpu
torch.cuda.manual_seed(SEED) #gpu
np.random.seed(SEED) #numpy
random.seed(SEED) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn


# Dataset
class SpetralSpatialDataset(tud.Dataset): # hdf5 format
    def __init__(self, data_dir):
        super(SpetralSpatialDataset, self).__init__()
        self.data_dir = data_dir
        self.length = len(h5py.File(self.data_dir, 'r')['targets'])

    def __getitem__(self, index):
        dataset = h5py.File(self.data_dir, 'r')

        spectrum, patch, target = (dataset['spectrums'][index, :],
                                   dataset['patches'][index,:,:,:].transpose(2,0,1),
                                   dataset['targets'][index])
        return spectrum, patch, target

    def __len__(self):
        return self.length

class unlabeled_Dataset(tud.Dataset):
    def __init__(self, u_sp, u_pat, u_target):
        super(unlabeled_Dataset, self).__init__()
        self.length = len(u_target)
        self.dataset = [u_sp, u_pat, u_target]

    def __getitem__(self, index):
        spectrum, patch, target = (self.dataset[0][index, :],
                                   self.dataset[1][index,:,:,:].transpose(2,0,1),
                                   self.dataset[2][index])
        return spectrum, patch, target

    def __len__(self):
        return self.length
# Networl

#  Conv block for resnet50
def Conv1(in_planes, places, stride=1):
    return nn.Sequential(
        #卷积核为7 * 7,stride = 2 padding为3
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=5,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        #最大池化层 3*3,stride =2
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        #维度扩张数
        self.expansion = expansion
        #是否降采用
        self.downsampling = downsampling
        #构建 图中各层的1*1,3*3,1*1的卷积块
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)
        #构成残差项
        out += residual
        out = self.relu(out)
        return out


def make_layer(in_places, places, block, stride, expansion = 4):
    layers = []
    #第1个
    layers.append(Bottleneck(in_places, places,stride, downsampling =True))
    for i in range(1, block):
        layers.append(Bottleneck(places*expansion, places))

    return nn.Sequential(*layers)

# Network
class SpetralSpatial(nn.Module):
    def __init__(self,time_step, pca_num, cls_num,):
        # output_shape = (image_shape-filter_shape+2*padding)/stride + 1
        super(SpetralSpatial, self).__init__()

        # spectral input: 4*32
        self.spectral_layer1 = nn.Sequential(
            nn.Conv1d(time_step, 64, 3, 1, padding = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, 1, padding = 1),
            nn.BatchNorm1d(64),
            #nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool1d(3, 1) # output: 64*30
        )

        self.spectral_layer2 = nn.Sequential(
            nn.Conv1d(64, 128, 3, 1, padding = 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, 1, padding = 1),
            nn.BatchNorm1d(128),
            #nn.Dropout(0.5),
            nn.ReLU(),
            nn.AvgPool1d(2, 2) # 128*15
        )

        self.spectral_fc1 = nn.Linear(1920, 128)
        #self.spectral_fc = nn.Linear(128, cls_num)

        # spatial
        self.expansion = 4

        self.conv1 = Conv1(in_planes = 4, places= 64)
        #层数
        self.layer1 = make_layer(in_places = 64, places= 64, block=3, stride=1)
        self.layer2 = make_layer(in_places = 256,places=128, block=4, stride=2)
        self.layer3 = make_layer(in_places=512,places=256, block=6, stride=2)
        self.layer4 = make_layer(in_places=1024,places=512, block=3, stride=2)
        #平均池化层
        self.avgpool = nn.AvgPool2d(3, stride=1)
        #全连接层
        self.spatial_fc1 = nn.Linear(2048,128)

        self.fc_joint = nn.Linear(256, 128)

        self.fc = nn.Linear(128, cls_num)


    def forward(self, x):
        # spectral
        x_sp = x[0]  #4*32

        x_sp = self.spectral_layer1(x_sp)
        x_sp = self.spectral_layer2(x_sp)
        x_sp = x_sp.view(-1, 1920)
        x_sp = F.relu(self.spectral_fc1(x_sp))
        out_sp = self.fc(x_sp)
        pred_sp = F.softmax(out_sp, dim = 1)

        # spatial
        x_pat = x[1] # 1*20*20
        x_pat = self.conv1(x_pat)

        x_pat = self.layer1(x_pat)
        x_pat = self.layer2(x_pat)
        x_pat = self.layer3(x_pat)
        x_pat = self.layer4(x_pat)

        x_pat = self.avgpool(x_pat)

        x_pat = x_pat.view(x_pat.size(0), -1)

        x_pat = self.spatial_fc1(x_pat)

        out_pat = self.fc(x_pat)
        pred_pat = F.softmax(out_pat, dim = 1)

        x_joint = self.fc_joint(torch.cat((x_sp,x_pat),dim = 1))
        out_joint = self.fc(x_joint)
        pred_joint = F.softmax(out_joint, dim = 1)

        return out_sp, out_pat, out_joint, pred_joint


def spectral_grp(spectrum, time_step, mode, device):
    N, bands = spectrum.shape
    slot = int(bands/time_step)
    gp_sp = torch.zeros(N, time_step, slot).to(device)
    if mode ==1:
        for j in range(time_step):
            gp_sp[:,j,:] = spectrum[:,j:j+(slot-1)*time_step+1:time_step]
    elif mode ==2:
        for j in range(time_step):
            gp_sp[:,j,:] = spectrum[:,j*slot:(j+1)*slot]

    return gp_sp


def pred_cat(pred):
    cat_pred = pred[0]
    for i in range(1,len(pred)):
        temp = pred[i]
        cat_pred = torch.cat((cat_pred, temp),dim = 0)

    return cat_pred


def pred_cat_np(pred):
    cat_pred = pred[0]
    for i in range(1,len(pred)):
        temp = pred[i]
        cat_pred = np.concatenate((cat_pred, temp),axis = 0)

    return cat_pred


def catList2Array(pred):
    cat_pred = pred[0]
    for i in range(1,len(pred)):
        temp = pred[i]
        cat_pred = torch.cat((cat_pred, temp),dim = 0)

    return cat_pred


def label_data(pred): # return label from svm pred after softmax
    labels = pred[0].max(dim=1)[1]
    for i in range(1,len(pred)):
        temp = pred[i].max(dim=1)[1]
        labels = torch.cat((labels, temp),dim = 0)

    return labels.numpy()+1


def evaluate_acc(model, dataloader, device):
    model.eval()
    pred_label = []
    ground_truth = []
    with torch.no_grad():
        for idx, (spectrums, patches, targets) in enumerate(dataloader):
            #sample_cp = copy.deepcopy(sample)
            spectrums, patches, targets = (spectrums.float().to(device),
                                           patches.float().to(device),
                                           targets.float().to(device))

            _, _, _, pred_joint = model([spectral_grp(spectrums,time_step, 2, device),
                                         patches])
            #_, _, _, pred_joint = model([spectrums,patches])

            pred_label.append(pred_joint.to('cpu').max(dim=1)[1]+1)
            ground_truth.append(targets.to('cpu'))

        all_pred = catList2Array(pred_label)
        all_gt = catList2Array(ground_truth)
        data_num = len(all_gt)
        tol_acc = ((all_gt == all_pred)*np.ones(data_num)).sum()/data_num

    return tol_acc


#supervised training function
def train(model, device, train_data, loss_fn, time_step, optimizer, epoch):
    model.train()
    for idx, (spectrums, patches, targets) in enumerate(train_data):
        #labels = targets
        spectrums, patches,targets = (spectrums.float().to(device),
                                      patches.float().to(device),
                                      targets.long().to(device))

        out_sp, out_pat, out_joint, pred_joint = model([spectral_grp(spectrums,time_step, 2, device),patches])
        #out_sp, out_pat, out_joint, pred_joint = model([spectrums,patches])
        loss = (loss_fn(out_sp, targets-1) + loss_fn(out_pat, targets-1) + loss_fn(out_joint, targets-1))

        #SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 1000== 0:
            #writer.add_scalar('training loss', loss.item(), epoch*len(train_data)+idx)
            print("Train Epoch: {}, iteration: {}, lr = {}, Loss: {}".format(
                epoch, idx, optimizer.param_groups[0]['lr'], loss.item()))


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(m.weight.data)

    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

    if isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    if isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    """if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()"""


# Semi supervised training
def train_comb(model, device, dataloader_l, dataloader_u, loss_fn, time_step, Cl, Cu, optimizer, epoch, lr):
    model.train()

    for idx, (data_l,data_u) in enumerate(zip(cycle(dataloader_l), dataloader_u)):

        spectrums_l, patches_l, targets_l = (data_l[0].float().to(device),
                                            data_l[1].float().to(device),
                                            data_l[2].long().to(device))

        spectrums_u, patches_u, targets_u = (data_u[0].float().to(device),
                                             data_u[1].float().to(device),
                                             data_u[2].long().to(device))

        (out_sp_l, out_pat_l,
         out_joint_l, pred_joint_l) = model([spectral_grp(spectrums_l,
                                                          time_step, 2, device),
                                             patches_l])

        (out_sp_u, out_pat_u,
         out_joint_u, pred_joint_u) = model([spectral_grp(spectrums_u,
                                                          time_step, 2, device),
                                             patches_u])

        loss = (Cl * loss_fn(out_sp_l,targets_l-1) + Cu * loss_fn(out_sp_u, targets_u-1) +
                Cl * loss_fn(out_pat_l,targets_l-1) + Cu * loss_fn(out_pat_u, targets_u-1) +
                Cl * loss_fn(out_joint_l,targets_l-1) + Cu * loss_fn(out_joint_u, targets_u-1))


        #Adam
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 3000 == 0:
            writer.add_scalar('training loss', loss.item(), epoch*(len(dataloader_l) + len(dataloader_u))+idx)
            print("Train Epoch: {}, iteration: {}, lr: {}, Loss: {}".format(
                epoch, idx, optimizer.param_groups[0]['lr'], loss.item()))


def evaluate(model, dataloader, device):
    model.eval()
    sp_out = []
    pat_out = []
    joint_out = []
    pred_out = []
    gt = []
    with torch.no_grad():
        for idx, (spectrums, patches, targets) in tqdm(enumerate(dataloader)):
            spectrums, patches, targets = (spectrums.float().to(device),
                                           patches.float().to(device),
                                           targets.to('cpu'))

            out_sp, out_pat, out_joint, pred_joint = model([spectral_grp(spectrums,
                                                                         time_step, 2, device),
                                                            patches])
            sp_out.append(out_sp.to('cpu'))
            pat_out.append(out_pat.to('cpu'))
            joint_out.append(out_joint.to('cpu'))
            pred_out.append(pred_joint.to('cpu'))
            gt.append(targets.numpy())

    return sp_out, pat_out, joint_out, pred_out, gt


def err_cal(joint_u, labels_u):
    num_labels = len(labels_u)
    joint_out = pred_cat(joint_u)
    corrects_joint = joint_out[range(num_labels), labels_u-1].unsqueeze(0).T
    margin = 1.0

    errs = (joint_out - corrects_joint + margin)

    return errs


def fake_label_eva(errs, pred_labels, labels_u, cls_num):
    new_fake_labels = labels_u.copy()
    #pred_labels = label_data(pred)
    update_flag = False
    update_num = 0
    for i in range(cls_num):
        cls_idx = np.where(labels_u == i+1)
        cls_pred = pred_labels[cls_idx]
        cls_members = errs[cls_idx]

        max_value , max_cls = cls_members.max(dim=1)

        count_dic  = Counter(max_cls.numpy())
        cls_count = list(count_dic.values()) # dictionary {cls-1:num}
        if len(cls_count) >= 2:
            max_labels_num = np.array(sorted(cls_count, reverse = True)[1])
            update_label_name = list(count_dic.keys())[cls_count.index(max_labels_num)]
            update_cls_idx = cls_idx[0][cls_pred == update_label_name+1]
            new_fake_labels[update_cls_idx] = update_label_name+1
            update_num += len(update_cls_idx)
            update_flag = True

        """for c in max_2_labels_nums:
            #if c > int(len(labels_u)/cls_num*0.005)
            update_label_name = list(count_dic.keys())[cls_count.index(c)]
            update_cls_idx = cls_idx[0][cls_pred == update_label_name+1]
            new_fake_labels[update_cls_idx] = update_label_name+1"""

    return new_fake_labels, update_flag, update_num


def updat_fake_label(label, file_name):
    f = h5py.File(file_name,'r+')
    f['targets'][()] = label
    f.close()


def unlabel_data_pack(dataset_u, temp_file_name, u_len):
    temp_dataloader = tud.DataLoader(dataset_u, batch_size = int(u_len),
                                     shuffle = False, num_workers =4)

    for _, (s, p, t) in enumerate(temp_dataloader):
        sp, pat, tar = s.float().numpy(), p.float().numpy(), t.byte().numpy()

    h =  h5py.File(temp_file_name, "w")
    h.close()

    h =  h5py.File(temp_file_name, "r+")

    pix_dst = h.create_dataset('spectrums', data = sp,chunks = (1,128))
    pat_dst = h.create_dataset('patches', data = pat.transpose(0,2,3,1), chunks = (1, 20, 20, 4))
    label_dst = h.create_dataset('targets', data = tar)
    h.close()

    return tar


"""Semi-Supervised Training"""

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    print('code runing, device is:', device)

    """data set initilizition """
    # load entire dataset
    dataset = SpetralSpatialDataset('hsi_crack_data_temp.hdf5')

    # split dataset into labeled and unlabeled
    tol_len = (dataset.length)
    ul_ratio = [0.04, 0.04, 0.92]
    split_ratio = np.array([0.99*ul_ratio[0], 0.99*ul_ratio[1], 0.99*ul_ratio[2], 0.01])
    u_len, l_len, empty_len, _ = np.ceil(tol_len*split_ratio)
    t_len = int(tol_len - u_len - l_len - empty_len)

    (dataset_u, _, dataset_l,
    test_dataset) = tud.random_split(
                                     dataset, [int(u_len), int(empty_len), int(l_len), t_len],
                                     generator = torch.Generator().manual_seed(SEED))

    dataloader_l = tud.DataLoader(dataset_l, batch_size = batch_size,
                                  shuffle = True, num_workers =4)

    test_dataloader = tud.DataLoader(test_dataset, batch_size = batch_size,
                                     shuffle = True, num_workers =4)

    # package unlabeled data into .hdf5 for further label update and reproduce dataloader
    temp_file_name = './temp_unlabel_dataset/unlabel_data004.hdf5'
    #unlabel_groudtruth = unlabel_data_pack(dataset_u, temp_file_name, u_len)

    """Training"""
    cls_num = 8
    pca_num = 4
    time_step = 4

    lr = 1e-4 # initial lr for supervised model
    acc_old = 0
    acc = 0
    train_epoch = 0


    writer = SummaryWriter('logs/Semi_train/data04_04/data0404_new_4') # torch board
    semi_model_name = './semi_train_models/data04_04/data0404_new_4.pth'
    print('Unlabel:Label Ratio is: {}, Batch Size is: {}, Initial lr is: {}, Save Model as {}'.format(
    ul_ratio, batch_size, lr, semi_model_name))

    # model build
    model = SpetralSpatial(time_step, pca_num, cls_num).to(device)
    model.apply(weights_init)
    #model = torch.load('./supervised_model_save/data100/data100_resnet50.h5')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    loss_fn = nn.MultiMarginLoss(p=1, margin = 1)  #, reduction = 'mean'

    # initial model only by labeled data
    train(model, device, dataloader_l, loss_fn, time_step, optimizer, 1)
    train_epoch +=1

    acc = evaluate_acc(model, test_dataloader,device)
    acc_old = acc

    writer.add_scalar('Test acc', acc.item(), train_epoch)

    print('Test Epoch:{}, lr = {}, test accuracy is:{}'.format(
       1, optimizer.param_groups[0]['lr'], acc))

    # create unlabeled dataloader
    dataset_fake = SpetralSpatialDataset(temp_file_name)
    dataloader_fake = tud.DataLoader(dataset_fake, batch_size = batch_size,
                                     shuffle = False, num_workers = 4)

    # produce fake label with supervised initial model
    labels_u = label_data(evaluate(model, dataloader_fake, device)[3])
    #tol_acc = ((unlabel_groudtruth == labels_u)*np.ones(int(u_len))).sum()/int(u_len)
    updat_fake_label(labels_u, temp_file_name)

    # semi training
    internal_epoch = 15
    lr = 1e-3
    adjust_learning_rate(optimizer, lr)
    acc_gate = 0.63
    update_flag = True
    update_num_old = 0

    Cl = 1.5
    Cu = 0.001
    train_time = 0

    while train_epoch < 300:
        train_time +=1
        for epoch in range(internal_epoch):
            if update_flag:
                del dataloader_fake
                dataset_fake = SpetralSpatialDataset(temp_file_name)
                dataloader_fake = tud.DataLoader(dataset_fake, batch_size = batch_size,
                                                 shuffle = False, num_workers = 4)

            train_epoch +=1
            train_comb(model, device, dataloader_l, dataloader_fake, loss_fn, time_step,
                       Cl, Cu, optimizer,train_epoch, lr)

            acc = evaluate_acc(model, test_dataloader,device)
            writer.add_scalar('Test acc', acc.item(), train_epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], train_epoch)

            print('Test Epoch:{}, lr = {}, test accuracy is:{}'.format(
               train_epoch, optimizer.param_groups[0]['lr'], acc))

            if acc > acc_old:
                if acc > acc_gate:
                    lr = lr/10
                    if acc_gate < 0.67:
                        acc_gate += 0.04
                    else:
                        acc_gate += 0.02
                    adjust_learning_rate(optimizer, lr)
                    print('acc crite changed......')
                    print('acc_gate = {}'.format(acc_gate))

                acc_old = acc
                # save model and optimized parameters
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }


                torch.save(state, semi_model_name)
                print('model saved as {}'.format(semi_model_name))

                model_improve = True
                train_time = 0
                break
            else:
                model_improve = False


        if train_time == 2: #and model_improve == False:
            print('model cannot improve during, load latest best model......')
            checkpoint = torch.load(semi_model_name)
            model.load_state_dict(checkpoint['model'])
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr = lr/10
            adjust_learning_rate(optimizer, lr)
            print('reduce lr to', lr)

            train_time = 0

        if model_improve: # adjust fake labels
            sp_u, pat_u, joint_u, pred, _ = evaluate(model, dataloader_fake, device)

            errs = err_cal(joint_u, labels_u)
            labels_u, update_flag, update_num = fake_label_eva(errs, label_data(pred), labels_u, cls_num)

            if update_flag: # and update_num > 100:
                if update_num > 300:
                    updat_fake_label(labels_u, temp_file_name)

                if update_num < 850:
                    Cu = min(2*Cu, Cl)
                    writer.add_scalar('Cu', Cu, train_epoch)
                print('Cu = {} now, {} unlabeled data labels are updated!'.format(Cu, update_num))
            else:
                Cu = min(2*Cu, Cl)
                writer.add_scalar('Cu', Cu, train_epoch)
                print('Cu = {} now, {} unlabeled data labels are updated!'.format(Cu, update_num))
