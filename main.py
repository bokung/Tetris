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
import copy

# Initial Global Parameters
ROWS = 20
COLUMNS = 10

# Size of template image matters, matchTemplate is just a 2d convolution, computing the difference using some function, it doesnt come with scaling capabilities unfortunately.
# To determine the full rectangle, take the dimensions of the board and add it to respective dimensions of the top left point of rectangle.
TEMPLATE_main_board = cv.imread('resources/main_board_no_garbage_template.png', cv.IMREAD_GRAYSCALE)
TEMPLATE_empty_black = cv.imread('resources/empty_black.png', cv.IMREAD_GRAYSCALE)
TEMPLATE_empty_white = cv.imread('resources/empty_white.png', cv.IMREAD_GRAYSCALE)
TEMPLATE_empty_low_quality = cv.imread('resources/empty_low_quality.png', cv.IMREAD_GRAYSCALE)
TEMPLATE_empty_low_quality_no_border = cv.imread('resources/empty_no_border.png', cv.IMREAD_GRAYSCALE)
TEMPLATE_block_light = cv.imread('resources/block_light.png', cv.IMREAD_GRAYSCALE)
TEMPLATE_block_medium = cv.imread('resources/block_medium.png', cv.IMREAD_GRAYSCALE)
TEMPLATE_block_dark = cv.imread('resources/block_dark.png', cv.IMREAD_GRAYSCALE)
TEMPLATE_preview = cv.imread('resources/preview_template.png', cv.IMREAD_GRAYSCALE)

test_duel = cv.imread('test/test_duel.png', cv.IMREAD_GRAYSCALE)
test_empty = cv.imread('test/test_empty.png', cv.IMREAD_GRAYSCALE)
test_lobby = cv.imread('test/test_lobby.png', cv.IMREAD_GRAYSCALE)
test_midgame = cv.imread('test/test_midgame.png', cv.IMREAD_GRAYSCALE)
test_block_detection = cv.imread('test/test_block_detection.png', cv.IMREAD_GRAYSCALE)

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
  cv.rectangle(board, top_left, bottom_right, color=(0, 0, 0), thickness=2)
  cv.imshow('Highlighted Board Position', board)
  # pyautogui.mouseInfo()
  cv.waitKey()

def lower_is_better_checker(match_template_mode):
  if (match_template_mode is cv.TM_SQDIFF or cv.TM_SQDIFF_NORMED):
    return True
  else:
    return False

def crop_board(top_left, bottom_right, fullscreen):
  cropped = fullscreen[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
  return cropped

def locate_player_board(template, fullscreen, confidence_threshold, template_match_mode):
  '''
  Use this to locate board when there are more than one board on the screen, e.g. in a 1v1 duel or a multiplayer lobby
  '''
  result = cv.matchTemplate(fullscreen, template, template_match_mode)
  lower_is_better = lower_is_better_checker(template_match_mode)

  # A lot of repeated code! Can cut down?
  if (lower_is_better):
    threshold = 1 - confidence_threshold
    boards = np.where(result <= threshold)
    threshold_increase = 0
    while (len(boards[0]) == 0):
      threshold_increase += 0.05
      boards = np.where(result <= threshold + threshold_increase)
    if (threshold_increase != 0):
      print('WARNING: Confidence threshold too high, automatically reducing confidence threshold by %s' %threshold_increase)
    if (1 - threshold - threshold_increase < 0.8):
      print('WARNING: Low confidence in board position')
  else:
    boards = np.where(result >= confidence_threshold)
    threshold = confidence_threshold
    threshold_decrease = 0
    while (len(boards[0]) == 0):
      threshold_decrease += 0.05
      boards = np.where(result >= confidence_threshold - threshold_decrease)
    if (threshold_decrease != 0):
      print('WARNING: Confidence threshold too high, automatically reducing confidence threshold by %s' %threshold_decrease)
    if (confidence_threshold - threshold_decrease < 0.9):
      print('WARNING: Low confidence in board position')
  
  min_x = np.min(boards[1]) # Take leftmost board, Player board is always on the left.
  results_along_min_x = result[:, min_x]
  best_y = np.argmin(results_along_min_x) if lower_is_better else np.argmax(results_along_min_x)
  top_left = (min_x, best_y)
  h = template.shape[0]
  w = template.shape[1]
  bottom_right = (min_x + w, best_y + h)
  return top_left, bottom_right

def highlight_squares(cropped_board, template):
  result = cv.matchTemplate(cropped_board, template, cv.TM_SQDIFF_NORMED)
  detected = list(zip(*np.where(result <= 0.2)[::-1]))
  h, w = template.shape
  for top_left in detected:
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv.rectangle(cropped_board, top_left, bottom_right, color=(255,0,0), thickness=1)
  # cv.imshow('cropped', cropped_board)
  # cv.waitKey()

def board_state(cropped_board):
  '''
  Only works when we do it on low settings! 
  Internally implemented as a template matching problem and we are using the low quality square block as reference.
  Returns a (ROW, COLUMN) ndarray of booleans, True means block is filled, False means block is empty.
  '''
  h, w = TEMPLATE_main_board.shape
  y_div = h/ROWS
  x_div = w/COLUMNS
  board = np.full((ROWS, COLUMNS), True) # Initialise to true, board detector will set detected empty squares to false.
  result = cv.matchTemplate(cropped_board, TEMPLATE_empty_low_quality_no_border, cv.TM_SQDIFF_NORMED)
  detected_empty = list(zip(*np.where(result <= 0.000001)[::-1])) # Returns a list of tuples of detected squares.
  for point in detected_empty:
    x, y = point
    r = int(y/y_div)
    c = int(x/x_div)
    board[r][c] = False
  return board

def visualise_board_state(board_state):
  h, w = TEMPLATE_main_board.shape
  y_div = h/ROWS
  x_div = w/COLUMNS
  img = np.zeros(TEMPLATE_main_board.shape)
  for r in range(ROWS):
    for c in range(COLUMNS):
      if (board_state[r, c]):
        img[int(r*y_div):int((r+1)*y_div-1), int(c*x_div):int((c+1)*x_div-1)] = 255
  return img


test_img = test_block_detection
top_left, bottom_right = locate_player_board(TEMPLATE_main_board, test_lobby, 0.9, cv.TM_SQDIFF_NORMED)
cropped = crop_board(top_left, bottom_right, test_img)
board_state = board_state(cropped)
print(board_state)
cv.imshow('Actual Board',cropped)
cv.imshow('What bot sees', visualise_board_state(board_state))
cv.waitKey()

# highlight_squares(cropped, TEMPLATE_empty_low_quality_no_border)
# cv.imshow('cropped', cropped)
# cv.waitKey()
# # highlight_squares(cropped, empty_black_template)
# # highlight_squares(cropped, empty_white_template)
# highlight_squares(cropped, block_light_template)
# highlight_squares(cropped, block_medium_template)
# highlight_squares(cropped, block_dark_template)
# highlight_squares(cropped, preview_template)
# cv.imshow('original', original)
# cv.imshow('colored', cropped)
# cv.waitKey()