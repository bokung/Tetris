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

# Size of template image matters, matchTemplate is just a 2d convolution, it doesnt come with scaling capabilities unfortunately.
# To determine the full rectangle, take the dimensions of the board and add it to respective dimensions of the top left point of rectangle.
main_board_template = cv.imread('main_board.png')
test_img = cv.imread('empty_board.png')

'''
Returns the top left point and bottom right points of the detected image as two tuples
'''
def locate_board(template, fullscreen):
  result = cv.matchTemplate(fullscreen, template, cv.TM_CCOEFF_NORMED)
  min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
  s = template.shape
  top_left = max_loc
  bottom_right = (max_loc[0] + s[1], max_loc[1] + s[0])
  return top_left, bottom_right

top_left, bottom_right = locate_board(main_board_template, test_img)
cv.rectangle(test_img, top_left, bottom_right, 255, 2)
cv.imshow('image', test_img)
cv.waitKey()

# print('Best match top left position: %s' % str(max_loc))
# print('Best match confidence: %s' % max_val)
# cv.imshow('main board', main_board_template)
# cv.imshow('full screen', test_img)
# cv.imshow('match', result)
# cv.waitKey()
# pyautogui.mouseInfo()