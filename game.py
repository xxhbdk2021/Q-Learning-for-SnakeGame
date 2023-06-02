# 实现手动游戏
# 1. 网格游戏背景(黑色), 方块蛇(蓝色), 食物(红色)
# 2. api_1 - 状态重置
# 3. api_2 - 单步游戏控制
#    input(action: left, forward-none, right)
#    output(overFlag, reward, rewardSum, foodCnt, iterCnt))

import time
from enum import Enum
from collections import namedtuple

import numpy
import pygame


Point = namedtuple("Point", ["colIdx", "rowIdx"])
color_black = (0, 0, 0)
color_white = (255, 255, 255)
color_red1 = (255, 0, 0)
color_green1 = (0, 255, 0)
color_blue1 = (0, 0, 255)


class Direction(Enum):

    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


class SnakeGame(object):
    '''
    reward design:
    1. eat food: +10
    2. game over: -10
    3. else: 0
    '''
    def __init__(self, rowNum=10, colNum=10, blockSize=15):
        self.__rowNum = rowNum
        self.__colNum = colNum
        self.__blockSize = blockSize
        self.__width = self.__colNum * self.__blockSize
        self.__height = self.__rowNum * self.__blockSize
        self.__dirs = [Direction.UP, Direction.LEFT, Direction.DOWN, Direction.RIGHT]
        self.__init_game()

        self.reset_game()

        self.__draw_background()
        self.__draw_snake()
        self.__draw_food()
        pygame.display.flip()


    def set_action(self, action, show_game=False, iterMax=1000):
        '''
        action:
        [1, 0, 0, 0] - up
        [0, 1, 0, 0] - left
        [0, 0, 1, 0] - down
        [0, 0, 0, 1] - right
        '''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        dirs = self.__dirs
        idxCurr = dirs.index(self.__direction)

        if numpy.array_equal(action, [1, 0, 0, 0]):
            idxNext = 0
        elif numpy.array_equal(action, [0, 1, 0, 0]):
            idxNext = 1
        elif numpy.array_equal(action, [0, 0, 1, 0]):
            idxNext = 2
        elif numpy.array_equal(action, [0, 0, 0, 1]):
            idxNext = 3

        self.__direction = dirs[idxNext]
        self.__move_snake()

        self.__iterCnt += 1
        reward = 0
        overStatus = False
        if self.__is_eaten():
            reward = 10
            self.__place_food()
            self.__foodCnt += 1
        elif self.__is_over():
            reward = -10
            overStatus = True
        self.__rewardSum += reward

        if not self.__is_over() and show_game:
            self.__draw_background()
            self.__draw_snake()
            self.__draw_food()
            pygame.display.flip()
            self.__clock.tick(10)

        if self.__iterCnt > iterMax:
            overStatus = True
        
        #####
        if overStatus:
            print(dirs)
            print(action)
            print(idxCurr, idxNext)
            print("dirCurr: ", dirs[idxCurr])
            print("dirNext: ", dirs[idxNext])
            print("reward: ", reward)
            print(self.__food)
            print(self.__snake)
        #####
        return overStatus, reward, self.__rewardSum, self.__foodCnt, self.__iterCnt


    def get_state(self):
        '''
        background: 0
        snake head: 1(channel 0)
        snake body: 1(channel 1)
        snake food: 1(channel 2)
        '''
        state = numpy.zeros((3, self.__rowNum+2, self.__colNum+2))
        head = self.__snake[0]
        body = self.__snake[1:]
        food = self.__food
        state[0, head.rowIdx+1, head.colIdx+1] = 1
        for point in body:
            state[1, point.rowIdx+1, point.colIdx+1] = 1
        state[2, food.rowIdx+1, food.colIdx+1] = 1
        return state


    def reset_game(self):
        self.__rewardSum = 0
        self.__foodCnt = 0
        self.__iterCnt = 0
        self.__init_snake()
        self.__place_food()


    def __is_eaten(self):
        head = self.__snake[0]
        eatFlag = head == self.__food
        return eatFlag


    def __is_over(self):
        head = self.__snake[0]
        body = self.__snake[1:]
        overFlag = False
        if head.colIdx < 0 or head.colIdx >= self.__colNum or \
            head.rowIdx < 0 or head.rowIdx >= self.__rowNum:
            overFlag = True
        elif head in body:
            overFlag = True

        return overFlag


    def __move_snake(self):
        head = self.__snake[0]
        colIdx = head.colIdx
        rowIdx = head.rowIdx
        if self.__direction == Direction.UP:
            rowIdx -= 1
        elif self.__direction == Direction.DOWN:
            rowIdx += 1
        elif self.__direction == Direction.LEFT:
            colIdx -= 1
        elif self.__direction == Direction.RIGHT:
            colIdx += 1
        head = Point(colIdx, rowIdx)
        self.__snake.insert(0, head)

        if not self.__is_eaten():
            self.__snake.pop()


    def __draw_food(self):
        pygame.draw.rect(self.__screen, color_red1, pygame.Rect(self.__food.colIdx*self.__blockSize+1,\
            self.__food.rowIdx*self.__blockSize+1, self.__blockSize-2, self.__blockSize-2))


    def __draw_snake(self):
        head = self.__snake[0]
        body = self.__snake[1:]
        pygame.draw.rect(self.__screen, color_green1, pygame.Rect(head.colIdx*self.__blockSize+1,\
            head.rowIdx*self.__blockSize+1, self.__blockSize-2, self.__blockSize-2))
        for point in body:
            pygame.draw.rect(self.__screen, color_blue1, pygame.Rect(point.colIdx*self.__blockSize+1,\
                point.rowIdx*self.__blockSize+1, self.__blockSize-2, self.__blockSize-2))


    def __place_food(self):
        colIdx = numpy.random.randint(0, self.__colNum)
        rowIdx = numpy.random.randint(0, self.__rowNum)
        self.__food = Point(colIdx, rowIdx)
        if self.__food in self.__snake:
            self.__place_food()


    # def __place_food(self):
    #     colIdx = numpy.random.randint(5, self.__colNum-5)
    #     rowIdx = numpy.random.randint(5, self.__rowNum-5)
    #     self.__food = Point(colIdx, rowIdx)
    #     if self.__food in self.__snake:
    #         self.__place_food()


    def __init_snake(self):
        self.__direction = Direction.RIGHT
        head = Point(self.__colNum//2, self.__rowNum//2)
        self.__snake = [
            head,
            Point(head.colIdx-1, head.rowIdx),
            Point(head.colIdx-2, head.rowIdx)
        ]


    def __draw_background(self):
        self.__screen.fill(color_black)
        for colIdx in range(self.__colNum+1):
            x = colIdx * self.__blockSize
            pygame.draw.line(self.__screen, color_white, (x, 0), (x, self.__height))
        for rowIdx in range(self.__rowNum+1):
            y = rowIdx * self.__blockSize
            pygame.draw.line(self.__screen, color_white, (0, y), (self.__width, y))
        text1 = self.__font.render(f"iterCnt: {self.__iterCnt}", True, color_white)
        text2 = self.__font.render(f"foodCnt: {self.__foodCnt}", True, color_white)
        self.__screen.blit(text1, [self.__blockSize, self.__blockSize])
        self.__screen.blit(text2, [self.__blockSize, self.__blockSize * 2])


    def __init_game(self):
        pygame.init()
        pygame.display.set_caption("snake game")
        self.__screen = pygame.display.set_mode((self.__width+1, self.__height+1))
        self.__font = pygame.font.Font("./assets/courierprime-1ovl.ttf", self.__blockSize)
        self.__clock = pygame.time.Clock()



# sgObj = SnakeGame(20, 20)
# time.sleep(5)
# state = sgObj.get_state()
# print(state)
# print(state.shape)

# time.sleep(60)

# while True:
#     action = numpy.array([0, 0, 0])
#     idx = numpy.random.randint(0, 3)
#     action[idx] = 1
    
#     overStatus, *_ = sgObj.set_action(action)
#     if overStatus:
#         sgObj.reset_game()