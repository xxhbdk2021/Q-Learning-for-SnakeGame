import os
import time

import torch
from torch import optim

from game import SnakeGame
from model import QNet, QLoss, train_epoch_model
from player import Player, DataStorage


# torch.manual_seed(1)

c = 10
lr = 0.0001
queSize = 10000
epochs = 1000
dsEpisode = 30
t = 300
batchSize = 100
modelname = "model.pt"

if os.path.isfile(modelname):
    modelObj = torch.load(modelname)
else:
    modelObj = QNet()
gameObj = SnakeGame(20, 20)
qlossObj = QLoss(modelObj, c=c)
optimObj = optim.Adam(qlossObj.parameters(), lr=lr)

playerObj = Player(modelObj)
dstoreObj = DataStorage(gameObj, queSize)


def train_model(dstoreObj, playerObj, qlossObj, optimObj, epochs, dsEpisode, t, batchSize):
    for epoch in range(epochs):
        playerObj.get_modelObj().train(False)
        foodAll = dstoreObj.save_buffer(playerObj, dsEpisode)

        playerObj.get_modelObj().train(True)
        for ti in range(t):
            batchData = dstoreObj.load_buffer(batchSize)
            loss = train_epoch_model(batchData, qlossObj, optimObj)
            print(f"epoch: {epoch}, foodAll: {foodAll}, ti: {ti}, loss: {loss:.6f}")
        
        torch.save(modelObj, "model.pt")


def test_model(gameObj, playerObj):
    gameObj.reset_game()
    playerObj.get_modelObj().train(False)
    
    while True:
        state = gameObj.get_state()
        playerObj.set_state(state)
        action = playerObj.get_action_qMax()
        overStatus, reward, rewardSum, foodCnt, iterCnt = gameObj.set_action(action, True, 10000)

        if overStatus:
            time.sleep(1)
            break

# test_model(gameObj, playerObj)
train_model(dstoreObj, playerObj, qlossObj, optimObj, epochs, dsEpisode, t, batchSize)