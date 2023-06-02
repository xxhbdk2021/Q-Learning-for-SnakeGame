import os
import copy

import numpy
import torch
from torch import nn
from torch import optim
from torch.utils import data


class QNet(nn.Module):
    
    def __init__(self):
        super(QNet, self).__init__()

        # 22 -> 10
        self.cov1 = nn.Conv2d(3, 128, 3, stride=1, padding=0, dtype=torch.float64)
        self.bn1 = nn.BatchNorm2d(128, dtype=torch.float64)
        # 10 -> 4
        self.cov2 = nn.Conv2d(128, 256, 3, stride=1, padding=0, dtype=torch.float64)
        self.bn2 = nn.BatchNorm2d(256, dtype=torch.float64)
        # 4 -> 1
        self.cov3 = nn.Conv2d(256, 512, 3, stride=1, padding=0, dtype=torch.float64)
        self.bn3 = nn.BatchNorm2d(512, dtype=torch.float64)
        # 512 -> 128
        self.lin4 = nn.Linear(512, 128, dtype=torch.float64)
        self.bn4 = nn.BatchNorm1d(128, dtype=torch.float64)
        # 128 -> 4
        self.lin5 = nn.Linear(128, 4, dtype=torch.float64)
        self.bn5 = nn.BatchNorm1d(4, dtype=torch.float64)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        

    def forward(self, X):
        X = self.pool(self.bn1(self.cov1(X)))
        X = self.pool(self.bn2(self.cov2(X)))
        X = self.pool(self.bn3(self.cov3(X)))
        X = self.relu(self.bn4(self.lin4(X.reshape(X.shape[:2]))))
        X = self.bn5(self.lin5(X))
        return X


class QLoss(nn.Module):

    def __init__(self, modelObj: QNet, c=10):
        super(QLoss, self).__init__()

        self.__modelObj = modelObj
        self.__modelObj_ = copy.deepcopy(modelObj)
        self.__c = c
        self.__iterCnt = 0


    def forward(self, batchData):
        if self.__iterCnt % self.__c == 0:
            self.__reset_model()
            self.__iterCnt = 0

        s0Tsr, a0Tsr, r0Tsr, s1Tsr = self.__parse_batchData(batchData)
        with torch.no_grad():
            q1Tsr = self.__modelObj_(s1Tsr)
            Y_ = (torch.max(q1Tsr, dim=1).values + r0Tsr).reshape((-1, 1))
        q0Tsr = self.__modelObj(s0Tsr)
        loss = torch.mean(a0Tsr * (Y_ - q0Tsr) ** 2)
        
        self.__iterCnt += 1
        return loss


    def __parse_batchData(self, batchData):
        s0Tsr, a0Tsr, r0Tsr, s1Tsr = list(), list(), list(), list()
        for data in batchData:
            s0Tsr.append(data[0])
            a0Tsr.append(data[1])
            r0Tsr.append(data[2])
            s1Tsr.append(data[3])

        s0Tsr = torch.from_numpy(numpy.array(s0Tsr))
        a0Tsr = torch.from_numpy(numpy.array(a0Tsr))
        r0Tsr = torch.from_numpy(numpy.array(r0Tsr))
        s1Tsr = torch.from_numpy(numpy.array(s1Tsr))
        return s0Tsr, a0Tsr, r0Tsr, s1Tsr


    def __reset_model(self):
        self.__modelObj_ = copy.deepcopy(self.__modelObj)


def train_epoch_model(batchData, qlossObj, optimObj):
    optimObj.zero_grad()
    loss = qlossObj(batchData)
    loss.backward()
    optimObj.step()
    return loss.item()



# X = torch.from_numpy(numpy.random.normal(0, 1, (10, 3, 22, 22)))

# QObj = QNet()
# X = QObj(X)
# print(X.shape)