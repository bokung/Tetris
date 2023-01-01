/*
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
'''
*/

// Kinda weird way to import stuff in a mainly functional language, libraries are treated as variables!
// Functions are exported explicitly and referenced in other files using some object! 
var robot = require('robotjs');

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function getMousePos() {
    while (true) {
        await sleep(1000); // wait 1 second
        console.log(robot.getMousePos());
    }
}

function humanMovement() {
    robot.moveMouseSmooth(31, 76);
}

// getMousePos();
humanMovement();