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

def locate_board(template, fullscreen):
  '''
  template: Template image to search for (Needle)
  fullscreen: Space to search for the template (Haystack)

  Returns the top left point and bottom right points of the detected image as two tuples
  '''

  result = cv.matchTemplate(fullscreen, template, cv.TM_CCOEFF_NORMED)
  min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
  s = template.shape
  top_left = max_loc
  bottom_right = (max_loc[0] + s[1], max_loc[1] + s[0])

  if (max_val < 0.8):
    print('WARNING: Low confidence in board position')

  return top_left, bottom_right

def highlight_board(top_left, bottom_right, board):
  '''
  Highlights the detected board location
  '''
  cv.rectangle(board, top_left, bottom_right, color=(0, 0, 255), thickness=2)
  cv.imshow('Highlighted Board Position', board)
  cv.waitKey()

def locate_player_board(template, fullscreen, confidence_threshold):
  '''
  Use this to locate board when there are more than one board on the screen, e.g. in a 1v1 duel or a multiplayer lobby
  '''
  result = cv.matchTemplate(fullscreen, template, cv.TM_CCOEFF_NORMED)
  boards = np.where(result >= confidence_threshold)
  player_board = np.min(boards, axis=0) # Take leftmost board, Player board is always on the left.
  w, h = template.shape
  top_left = player_board
  bottom_right = (top_left[0] + h, top_left[0] + w)
  
  return top_left, bottom_right


# top_left, bottom_right = locate_board(main_board_template, test_img)
# highlight_board(top_left, bottom_right, test_img)

# print('Best match top left position: %s' % str(max_loc))
# print('Best match confidence: %s' % max_val)
# cv.imshow('main board', main_board_template)
# cv.imshow('full screen', test_img)
# cv.imshow('match', result)
# cv.waitKey()
# pyautogui.mouseInfo()