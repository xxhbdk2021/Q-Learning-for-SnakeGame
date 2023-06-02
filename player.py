import numpy
import torch


class Player(object):

    def __init__(self, modelObj):
        self.__modelObj = modelObj


    def set_state(self, state: numpy.ndarray):
        state = torch.from_numpy(state)
        self.__state = state.reshape((1, *state.shape))
        with torch.no_grad():
            qVal = self.__modelObj(self.__state)
            self.__qVal = qVal.numpy()


    def get_action_qMax(self):
        '''
        a = argmax Q
        '''
        qVal = self.__qVal
        idx = numpy.argmax(qVal[0])
        action = numpy.zeros((qVal.shape[1], )).astype(numpy.int64)
        action[idx] = 1
        return action


    def get_action_boltzmann(self):
        '''
        Boltzmann Exploration
        '''
        qVal = self.__qVal

        term1 = numpy.exp(qVal)
        term2 = term1 / numpy.sum(term1)
        term3 = numpy.cumsum(term2)
        randNum = numpy.random.uniform(0, 1)

        idx = 0
        while randNum >= term3[idx]:
            idx += 1

        action = numpy.zeros((qVal.shape[1], )).astype(numpy.int64)
        action[idx] = 1
        return action


    def get_modelObj(self):
        return self.__modelObj


class DataStorage(object):

    def __init__(self, gameObj, queSize=1000):
        self.__gameObj = gameObj
        self.__queSize = queSize
        self.__queData = list()


    def load_buffer(self, batchSize=30):
        queSize = len(self.__queData)
        randIdxList = numpy.random.randint(0, queSize, batchSize)
        batchData = list(self.__queData[idx] for idx in randIdxList)
        return batchData


    def save_buffer(self, playerObj, episode=10):
        foodAll = 0
        for i in range(episode):
            state0 = self.__gameObj.get_state()
            while True:
                playerObj.set_state(state0)
                action0 = playerObj.get_action_boltzmann()
                overStatus, reward0, rewardSum, foodCnt, iterCnt = self.__gameObj.set_action(action0)
                state1 = self.__gameObj.get_state()
                self.__append_data(state0, action0, reward0, state1)
                if overStatus:
                    foodAll += foodCnt
                    self.__gameObj.reset_game()
                    break
                state0 = state1
        return foodAll


    def __append_data(self, s0, a0, r0, s1):
        if len(self.__queData) >= self.__queSize:
            self.__queData.pop(0)
        data = (s0, a0, r0, s1)
        self.__queData.append(data)


    def __len__(self):
        return len(self.__queData)
        
    
    
        


# gameObj = SnakeGame(20, 20)
# modelObj = QNet()
# playerObj = Player(modelObj)

# dstoreObj = DataStorage(gameObj, 1000)
# dstoreObj.save_buffer(playerObj, 1)
# batchData = dstoreObj.load_buffer(5)

# qlossObj = QLoss(modelObj)
# qlossObj(batchData)