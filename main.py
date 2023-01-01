'''
First Problem: How to read input from Tetr.io and convert it to a usable tetris board in our local machine?
Possible Approaches:
> Somehow read web elements directly?
> Keep capturing the screen at every frame and use an algorithm to convert it to a usable representation?
> Use Tetrio api (Is there even one?)
> Turn on audio cues in settings at use sounds to determine next piece?
> Read web socket packages and somehow extract data about board state (how to do?)
> Download desktop version and read memory locations that give current board state

Interface input(int row, int col) {
    board => np.array((row, col)) # board[i][j] = 0 if its an empty space, 1 if it has a block. We don't need information about colour of every square.
    hold = some tetris piece that we are holding, initialise to null
    next = LinkedList, everytime we put down a piece we append next piece to end of list, and set new head to be the next element.
}

Working Methods:
misterhat: Set fixed sized window and read colours of pixels at predefined locations (hard coded, BAD!)
           > Minor improvements: take screenshots but be able to determine where main board is without hard coding, find some relation between play area position and window size.
             > Robotjs is too limited in functionality, hard to actually develop anything remotely general and robust.

Confirmed direction:
> Use OpenCV as input manager to create a resilient tetrio bot that doesn't break whenever the tetrio devs change board size, we change the window size or some other small thing.

NO HARDCODING!
'''

import cv2 as cv
import numpy as np
import pyautogui

main_board_img = cv.imread('main_board.png')
test_img = cv.imread('test_smaller_window.png')
result = cv.matchTemplate(test_img, main_board_img, cv.TM_CCOEFF)

# cv.imshow('main board', main_board_img)
# cv.imshow('full screen', test_img)
# cv.imshow('match', result)
# cv.waitKey()