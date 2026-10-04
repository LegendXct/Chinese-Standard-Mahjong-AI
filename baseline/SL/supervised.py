from dataset import MahjongGBDataset
from torch.utils.data import DataLoader
import torch.nn as nn
#from model import CNNModel
import torch.nn.functional as F
import torch
import os

class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, use_1x1conv=False, strides=1) :
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm1d(output_channels)
        self.bn2 = nn.BatchNorm1d(output_channels)
    
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        if self.conv3:
            x = self.conv3(x)
        y = y + x
        return y
    
class BottleNeck(nn.Module):
    def __init__(self, input_channels, output_channels, use_1x1conv=False, strides=1) :
        super().__init__()
        mid_channels = input_channels
        if input_channels / 4 != 0:
            mid_channels = int(mid_channels / 4)
        self.conv1 = nn.Conv1d(input_channels, mid_channels, kernel_size=1, padding=0, stride=strides)
        self.conv2 = nn.Conv1d(mid_channels, mid_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(mid_channels, output_channels, kernel_size=1, padding=0, stride=strides)
        if use_1x1conv:
            self.conv4 = nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=strides)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.bn2 = nn.BatchNorm1d(mid_channels)
        self.bn3 = nn.BatchNorm1d(output_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = F.relu(self.bn3(self.conv3(y)))
        if self.conv4:
            x = self.conv4(x)
        y = y + x
        return y

class MyResNet(nn.Module):

    def __init__(self) :
        super().__init__()
        self.bl1 = nn.Sequential(nn.Conv1d(141, 256, kernel_size=3, stride=1, padding=1),
                                 nn.BatchNorm1d(256), nn.ReLU()
                                 )
        self.bl2 = nn.Sequential(*self.res_layer_maker(256, 256, 5, first_block=True))
        self.bl3 = nn.Sequential(*self.res_layer_maker(256, 512, 5))
        self.bl4 = nn.Sequential(*self.res_layer_maker(512, 1024, 5))
        self.bottleneck = BottleNeck(1024,1024)
        self.bl5 = nn.Sequential(*self.res_layer_maker(1024,1024,5))
        #self.pool1
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1024*5, 44)

    def forward(self, x):
        x = self.bl1(x)
        x = self.bl2(x)
        x = self.bl3(x)
        x = self.bl4(x)
        x = self.bottleneck(x)
        x = self.bl5(x)
        x = self.linear(self.flatten(x))
        return F.log_softmax(x, dim=1)

    @staticmethod
    def res_layer_maker(input_channels, output_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(ResBlock(input_channels, output_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(ResBlock(output_channels, output_channels))
        return blk

if __name__ == '__main__':
    logdir = 'model/'
    os.mkdir(logdir + 'checkpoint')
    
    # Load dataset
    splitRatio = 0.9
    batchSize = 1024
    trainDataset = MahjongGBDataset(0, splitRatio, True)
    validateDataset = MahjongGBDataset(splitRatio, 1, False)
    loader = DataLoader(dataset = trainDataset, batch_size = batchSize, shuffle = True)
    vloader = DataLoader(dataset = validateDataset, batch_size = batchSize, shuffle = False)
    

    # Load model
    model = MyResNet().to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4)
    
    # Train and validate
    for e in range(20):
        print('Epoch', e)
        torch.save(model.state_dict(), logdir + 'checkpoint/%d.pkl' % e)
        for i, d in enumerate(loader):
            input_dict = {'is_training': True, 'obs': {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}}
            logits = model(input_dict)
            loss = F.cross_entropy(logits, d[2].long().cuda())
            if i % 128 == 0:
                print('Iteration %d/%d'%(i, len(trainDataset) // batchSize + 1), 'policy_loss', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Run validation:')
        correct = 0
        for i, d in enumerate(vloader):
            input_dict = {'is_training': False, 'obs': {'observation': d[0].cuda(), 'action_mask': d[1].cuda()}}
            with torch.no_grad():
                logits = model(input_dict)
                pred = logits.argmax(dim = 1)
                correct += torch.eq(pred, d[2].cuda()).sum().item()
        acc = correct / len(validateDataset)
        print('Epoch', e + 1, 'Validate acc:', acc)