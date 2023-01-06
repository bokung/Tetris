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

class Detector:

  def __init__(self, ROWS=20, COLUMNS=10) -> None: # Default board size to 20x10
    self.ROWS = ROWS
    self.COLUMNS = COLUMNS

  # Size of template image matters, matchTemplate is just a 2d convolution, computing the difference using some function, it doesnt come with scaling capabilities unfortunately.
  # To determine the full rectangle, take the dimensions of the board and add it to respective dimensions of the top left point of rectangle.
  TEMPLATE_main_board = cv.imread('resources/main_board_with_border.png', cv.IMREAD_GRAYSCALE)
  TEMPLATE_empty_black = cv.imread('resources/empty_black.png', cv.IMREAD_GRAYSCALE)
  TEMPLATE_empty_white = cv.imread('resources/empty_white.png', cv.IMREAD_GRAYSCALE)
  TEMPLATE_empty_low_quality = cv.imread('resources/empty_low_quality.png', cv.IMREAD_GRAYSCALE)
  TEMPLATE_empty_low_quality_no_border = cv.imread('resources/empty_no_border.png', cv.IMREAD_GRAYSCALE)
  TEMPLATE_block_light = cv.imread('resources/block_light.png', cv.IMREAD_GRAYSCALE)
  TEMPLATE_block_medium = cv.imread('resources/block_medium.png', cv.IMREAD_GRAYSCALE)
  TEMPLATE_block_dark = cv.imread('resources/block_dark.png', cv.IMREAD_GRAYSCALE)
  TEMPLATE_preview = cv.imread('resources/preview_template.png', cv.IMREAD_GRAYSCALE)

  MASK_main_board = np.full(TEMPLATE_main_board.shape, fill_value=255, dtype="uint8")
  cv.rectangle(MASK_main_board, (0, 5), (466, 921), 0, -1)

  test_duel = cv.imread('test/test_duel.png', cv.IMREAD_GRAYSCALE)
  test_empty = cv.imread('test/test_empty.png', cv.IMREAD_GRAYSCALE)
  test_empty_low = cv.imread('test/test_empty_low.png', cv.IMREAD_GRAYSCALE)
  test_lobby = cv.imread('test/test_lobby.png', cv.IMREAD_GRAYSCALE)
  test_midgame = cv.imread('test/test_midgame.png', cv.IMREAD_GRAYSCALE)
  test_block_detection = cv.imread('test/test_block_detection.png', cv.IMREAD_GRAYSCALE)
  test_zen = cv.imread('test/test_zen.png', cv.IMREAD_GRAYSCALE)

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

  @staticmethod
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
    result = cv.matchTemplate(fullscreen, template, template_match_mode, mask=Detector.MASK_main_board)
    lower_is_better = Detector.lower_is_better_checker(template_match_mode)

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

  def board_state(self, cropped_board):
    '''
    Only works when we do it on low settings! 
    Internally implemented as a template matching problem and we are using the low quality square block as reference.
    Returns a (ROW, COLUMN) ndarray of booleans, True means block is filled, False means block is empty.
    '''
    h, w = Detector.TEMPLATE_main_board.shape
    y_div = h/self.ROWS
    x_div = w/self.COLUMNS
    board = np.full((self.ROWS, self.COLUMNS), True) # Initialise to true, board detector will set detected empty squares to false.
    result = cv.matchTemplate(cropped_board, Detector.TEMPLATE_empty_low_quality_no_border, cv.TM_CCOEFF_NORMED)
    # detected_empty = list(zip(*np.where(result <= 0.000001)[::-1])) # Returns a list of tuples of detected squares, for Squared difference comparison
    detected_empty = list(zip(*np.where(result >= 0.99)[::-1]))
    for point in detected_empty:
      x, y = point
      # r = min(round(y/y_div), ROWS - 1)
      # c = min(round(x/x_div), COLUMNS - 1)
      r = round(y/y_div)
      c = round(x/x_div)
      
      if (r <= -1 or r >= 20 or c <= -1 or c >= 10):
        print("BAD THINGS HAPPENING!!!")

      board[r, c] = False
    return board

  def visualise_board_state(self, board_state):
    h, w = Detector.TEMPLATE_main_board.shape
    y_div = h/self.ROWS
    x_div = w/self.COLUMNS
    img = np.zeros(Detector.TEMPLATE_main_board.shape)
    for r in range(self.ROWS):
      for c in range(self.COLUMNS):
        if (board_state[r, c]):
          img[int(r*y_div):int((r+1)*y_div-1), int(c*x_div):int((c+1)*x_div-1)] = 255
    return img

  def real_time_state_detection(self):
    # Run once to detect board location at the start of the game when we have an empty board
    screenshot = pyautogui.screenshot()
    fullscreen = np.array(screenshot)
    fullscreen = cv.cvtColor(fullscreen, cv.COLOR_RGB2GRAY)
    top_left, bottom_right = Detector.locate_player_board(Detector.TEMPLATE_main_board, fullscreen, 0.99, cv.TM_SQDIFF_NORMED)

    while (True):
      screenshot = pyautogui.screenshot()
      fullscreen = np.array(screenshot) # Default behaviour is a RGB picture for pyautogui screen capture. Images in OpenCV are stored in BGR format.
      fullscreen = cv.cvtColor(fullscreen, cv.COLOR_RGB2GRAY)
      cropped = Detector.crop_board(top_left, bottom_right, fullscreen)
      board = Detector.board_state(self, cropped)
      board_visualisation = Detector.visualise_board_state(self, board)
      cv.imshow('Original Board', cropped)
      cv.imshow('Detected Board State', board_visualisation)
      if cv.waitKey(1) == ord('q'): # Wait 1ms, press q to quit
        cv.destroyAllWindows()
        break

detector = Detector()
detector.real_time_state_detection()
