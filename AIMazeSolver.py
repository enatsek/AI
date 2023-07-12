#!/usr/bin/env python3
'''

AI Maze solver
Applies Q learning algorithms for learning the shortest path of each cell of a maze.
Maze size could be from 3*3 to 26*26.
Mazes are created randomly.
Learning takes a lot of time for the bigger mazes, so multiprocessing is used.
numpy library is used, so it must be installed:
pip3 install numpy
All other libraries are standard Python libraries

Maze creation parts are taken from https://rosettacode.org/wiki/Maze_generation#Python

Learning part including Rewards and Q array calculations are taken from book:
AI Crash Course by Hadelin de Ponteves from Packt Publications
and the book's github page: 

'''
# Importing the libraries
# numpy has to be installed by "pip3 install numpy"
import numpy as np                      # numpy arrays are used
from random import shuffle, randrange   # Used for random maze
import sys                              # Used for direct printing to the screen
import time                             # Used for calculation elapsed learning time
import multiprocessing                  # Used for multiprocessing at the learning phase


# Parameters gamma and alpha for the Q-Learning
gamma = 0.75
alpha = 0.9

# Global variables
ncpus = multiprocessing.cpu_count()     # Number of CPUs on the computer
nprocesses = ncpus - 1      # Number of parallel processes, optimum value would be 1 less than total CPUs

if nprocesses == 0:
    nprocesses = 1 

MAX_REWARD = 1000           # Reward point for finding a cell

menu_options = ["0", "1", "2", "3", "4", "5", "8", "9"]


min_dim = 3                 # Minimum width and height for the maze
max_dim = 26                # Maximum width and height for the maze
maze_width = 4              # max 26
maze_height = 4             # max 26
maze_len = maze_width * maze_height   # Total cells in maze
N = maze_len * 1000          # Learning count, set as maze_len * 1000
location_to_state = {}      # Location to state conversion aa -> 0 ab -> 1
state_to_location = {}      # State to location conversion 0 -> aa  1-> ab
maze = []                   # Produced maze

def get_location_state():
# Returns location_to_state and state_to_location dictionaries
    location_to_state = {}
    for i in range(maze_height):
        for j in range(maze_width):
            location_to_state[chr(ord('a') + i) + chr(ord('a') + j)] = i * maze_width + j

    # Making a mapping from the states to the locations
    state_to_location = {state: location for location, state in location_to_state.items()}

    return(location_to_state, state_to_location)


# Random Maze Creation
def make_maze(w, h):
# Create a random maze with given width and height
# from https://rosettacode.org/wiki/Maze_generation#Python
# 
# Returns maze as a list array with printable ascii chars
    vis = [[0] * w + [1] for _ in range(h)] + [[1] * (w + 1)]
    ver = [["|  "] * w + ['|'] for _ in range(h)] + [[]]
    hor = [["+--"] * w + ['+'] for _ in range(h + 1)]

    maze = []

    def walk(x, y):
        vis[y][x] = 1

        d = [(x - 1, y), (x, y + 1), (x + 1, y), (x, y - 1)]
        shuffle(d)
        for (xx, yy) in d:
            if vis[yy][xx]: continue
            if xx == x: hor[max(y, yy)][x] = "+  "
            if yy == y: ver[y][max(x, xx)] = "   "
            walk(xx, yy)

    walk(randrange(w), randrange(h))
    for (a, b) in zip(hor, ver):
        maze.append(''.join(a + ['\n'] + b))
    return(maze)

def print_maze(maze):
# Prints given maze
    for line in maze:
        print(line)

def print_maze_wlabels(maze):
# Prints given maze with labels in each cell
# Each row starts with the next letter
# aa ab ac ad ...
# ba bb bc bd ...
# ca cb cc cd ...
# ...

    for i in (range(maze_height + 1)):
        # Print the first line (includes top borders, bottom borders for the last line)
        maze_line = maze[i][0:maze_width*3+2]
        # If not in bottom line
        if i != maze_height:
            for j in range(maze_width):
                maze_line += maze[i][maze_width*3 +2 + j*3] + chr(ord('a') + i) + chr(ord('a') + j) 
            maze_line += maze[i][maze_width*3 +2 + (j+1)*3]
        print(maze_line)


def get_maze_props(maze):
# Returns a 2D array, cells contain information about the
#   corresponding maze cell.
# Information consists if the cell's connection to 
#    North, East, South, or Westcell is open
# North -> 1st bit, 1 open, 0 closed
# East  -> 2nd bit, 1 open, 0 closed
# South -> 3rd bit, 1 open, 0 closed
# West  -> 4th bit, 1 open, 0 closed
# 0  -> all closed
# 15 -> all open
# 10 -> West open, South closed, East open, North closed
#
# Returns a 2 dimensional Numpy array, with the default maze width and height
    maze_props = np.array(np.zeros([maze_height, maze_width]))
    for i in range(maze_height):
        for j in range(maze_width):
            # Check North cell
            if maze[i][j*3 + 1] == ' ':
                maze_props[i][j] += 1  # North is open
            # Check East cell
            if maze[i][maze_width*3 +2 + j*3+ 3] == ' ':
                maze_props[i][j] += 2  # East is open
            # Check South cell
            if maze[i+1][j*3 + 1] == ' ':
                maze_props[i][j] += 4  # South is open
            # Check West cell
            if maze[i][maze_width*3 +2 + j*3] == ' ':
                maze_props[i][j] += 8  # West is open
    return(maze_props)


def get_raw_rewards(maze):
# Returns a 2D Numpy array with both maze_len dimensions
# For each line, it contains 1 for the cells that are movable, 0 otherwise
# Returns the Numpy array, with the rewards for a specific cell
# This function is called for every cell of the maze

    # Get properties for the maze
    maze_props = get_maze_props(maze)
    # Defining the raw rewards
    # Start with full zeros
    RR = np.zeros((maze_len, maze_len))
    # Assign 1 reward to each cell's moveable cells
    for i in range(maze_len):
        # maze_len does not have width or height info, we need to calculate
        h = i // maze_width
        w = i % maze_width
        prop = int(maze_props[h][w])
        # Check if north is open
        if (prop % 2):
            # No North neighbor for the first line
            if (h > 0):
                # Set the reward of the North neighbor as 1
                RR[i][i-maze_width] = 1
        # Shift prop right, last bit is East now
        prop = prop >> 1
        # Check if east is open
        if (prop % 2):
            # No East neighbor for the last column
            if (w != (maze_width - 1)):
                # Set the reward of the East neighbor as 1
                RR[i][i+1] = 1
        # Shift prop right, last bit is South now
        prop = prop >> 1
        # Check if south is open
        if (prop % 2):
            # No South neighbor for the last line
            if (h != maze_height -1):
                # Set the reward of the South neighbor as 1
                RR[i][i+maze_width] = 1
        # Shift prop right, last bit is West now
        prop = prop >> 1
        # Check if west is open
        if (prop % 2):
            # No West neighbor for the first column
            if (w > 0):
                # Set the reward of the East neighbor as 1
                RR[i][i-1] = 1
    return(RR)

def get_full_rewards(maze):
# Returns a 3D Numpy array for all maze_len dimensions
# Includes reward array for all cells.
# First dimension is the cell number, 2nd and 3rd dimensions are the reward array.
# Reward arrays contain MAX_REWARD for the cell itself, 1 for the cells that are movable, 0 otherwise
# Returns the Numpy array
    RR = get_raw_rewards(maze)
    FR = np.zeros((maze_len, maze_len, maze_len))
    for i in range(maze_len):
        # Gets rewards for each cell
        FR[i] = RR
    for i in range(maze_len):
        # Set the cell's own reward to MAX_REWARD
        FR[i][i][i] = MAX_REWARD
    return(FR)


def learn_one(R, num):
# Calculates a Q array (q) for a maze's cell
# This function must be called for all the cells of the maze
# Gets reward array (as R), which is a 2D Numpy array of dimensions maze_len * maze_len
# num parameter is used to return info to identify calling index number. This function is called
#   by many processes, so we need a track to identify each call.
# q array is a 2D Numpy array of the same dimensions as R (maze_len * maze_len)
# Returns num and q array
    # Apply Q array algorithm
    q = np.array(np.zeros([maze_len, maze_len]))
    for i in range(N):
        current_state = np.random.randint(0,maze_len)
        playable_actions = []
        for j in range(maze_len):
            if R[current_state, j] > 0:
                playable_actions.append(j)
        next_state = np.random.choice(playable_actions)
        TD = R[current_state, next_state] + gamma * q[next_state, np.argmax(q[next_state,])] - q[current_state, next_state]
        q[current_state, next_state] = q[current_state, next_state] + alpha * TD
    # Display info
    sys.stdout.write(f"\b\b\b\b" + str(num))
    sys.stdout.flush()
    return(num, q)

def learn_all(F, quite = False):
# Calculates a Q array for all cells of the maze
# Gets full reward array (as F), which is a 3D Numpy array of dimensions maze_len * maze_len * maze_len
# Q array is a 3D Numpy array of the same dimensions as F (maze_len * maze_len)
# Returns q array
    start_time = time.time()
    Q = np.array(np.zeros([maze_len, maze_len, maze_len]))
    print("Learning. Total", maze_len, "steps.")

    # Create a multiprocessing pool with nprocess number of processes
    pool = multiprocessing.Pool(processes=nprocesses)

    # Apply learn_one function for all the cells asynchronously
    results = []
    for i in range(maze_len):
        result = pool.apply_async(learn_one, (F[i], i))
        results.append(result)

    # Get the results
    for result in results:
        i, arr = result.get()
        Q[i] = arr
    # Close the pool
    pool.close()
    pool.join()

    sys.stdout.write(f"\b\b\b\b" + "Learning Complete")
    sys.stdout.flush()
    print()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Took", elapsed_time, "seconds.")
    return(Q)
    


def get_route(Q, starting_location, ending_location):
# Returns the route from start to end as numbers in a list
# Q is the Q array for the end cell, start and end are the cell labels (aa, ab, ac ..) of the maze.
# Starts from the start and moves to the neighbor cell with the highest Q value, until reaching the end.
    route = [starting_location]
    next_location = starting_location
    while (next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state,])
        temp_next_location = next_location
        next_location = state_to_location[next_state]
        # Route lenght cannot be more than maze_len. We are stuck somewhere.
        if (len(route) > maze_len):
            route = []
            break
        route.append(next_location)
        starting_location = next_location
    return route

def print_route(maze, route):
# Prints the maze with the route found by the AI
# route has the labels of the cells (aa, ab, ac, ba ...) to fill
# Similar to print_maze_wlabels, just prints labels for the cells in the route list
    # Create a new route_number, with cell numbers from the original route
    route_numbers = []
    for cell in route:
        route_numbers.append(location_to_state[cell])
    for i in (range(maze_height + 1)):
        maze_line = maze[i][0:maze_width*3+2]
        if i != maze_height:
            for j in range(maze_width):
                maze_line += maze[i][maze_width*3 +2 + j*3] 
                if (i*maze_height+j) in route_numbers:
                    maze_line += chr(ord('a') + i) + chr(ord('a') + j) 
                else:
                    maze_line += "  "
            maze_line += maze[i][maze_width*3 +2 + (j+1)*3]
        print(maze_line)


def get_maze_dimensions():
# Asks the user to enter maze width and maze height
# If the user press enter for the width or height, returns 0
# Otherwise returns entered width and height
    got_dimensions = False
    while (not got_dimensions):
        print("Enter the new maze width and height, separated with a space.")
        print(" Min:", min_dim, "Max:", max_dim, ". Press just enter to cancel.")
        str_arr = input().split()
        # Return 0 ,f the selection is empty
        if len(str_arr) == 0:
            width = height = 0
            got_dimensions = True
        elif len(str_arr) == 2:
            if (str_arr[0].isdigit() and str_arr[1].isdigit()):
                width = int(str_arr[0])
                height = int(str_arr[1])
                if (width>=min_dim and width<= max_dim and height >= min_dim and height <= max_dim):
                    got_dimensions = True
    return(width, height)

def get_start_end():
# Asks the user to enter the start and end positions to find a route
# If the user just press enter returns empty strings
# Otherwise returns entered start and end
    got_cells = False
    while (not got_cells):
        print()
        print("AI is going to find the route between start and end")
        print("Enter the start and end cells separated with a space.")
        print("Enter cells like aa ca. Press just enter to cancel.")
        str_arr = input().split()
        # Return 0 ,f the selection is empty
        if len(str_arr) == 0:
            start = end = 0
            got_cells = True
        elif len(str_arr) == 2:
            start = str_arr[0]
            end = str_arr[1]
            if ((start in location_to_state) and (end in location_to_state)):
                got_cells = True
    return(start, end)

def get_number_of_processes():
# Asks the user to enter the number of parallel processes
# If the user press enter returns 0
# Otherwise returns entered number
    got_processes = False
    while (not got_processes):
        print("Enter the number of parallel processes.")
        print("This computer has", ncpus, "CPU core.")
        str_arr = input().split()
        # Return 0 ,f the selection is empty
        if len(str_arr) == 0:
            processes = 0
            got_processes = True
        elif len(str_arr) == 1:
            if (str_arr[0].isdigit()):
                processes = int(str_arr[0])
                if (processes > 0):
                    got_processes = True
    return(processes)


def initialize():
# Set global parameters, produce the first maze
# Called at start and after maze dimension changes
    global maze_len
    global location_to_state
    global state_to_location
    global maze
    global N

    maze_len = maze_height * maze_width
    N = maze_len * 1000
    location_to_state, state_to_location = get_location_state()
    # Create a random maze
    maze = make_maze(maze_width, maze_height)
    print_maze(maze)

def menu():
# Displays the menu, returns the selection
    print()
    print("1 - Create a new maze.")
    print("2 - Print the current maze.")
    print("3 - Print the current maze with labels.")
    print("4 - Let the AI learn the maze. Current:", learned)
    print("5 - Ask AI to find a route in the maze (AI must learn first).")
    print("  -")
    print("8 - Change maze dimensions. Current H:", maze_height, "W:", maze_width)
    print("9 - Change number of parallel processes. Current:", nprocesses)
    print("0 - Exit.")
    option = ""
    while (option.strip() not in menu_options):
        option = input("Select an option: ")
    return(int(option.strip()))



learned = False         # Flag if the current maze is learned
initialize()
selection = 1
while selection != 0:
    selection = menu()
    # Create a new maze
    if selection == 1:
        maze = make_maze(maze_width, maze_height)
        print_maze(maze)
        learned = False
    # Print the maze
    elif selection == 2:
        print_maze(maze)
    # Print the maze with the labels
    elif selection == 3:
        print_maze_wlabels(maze)
    # Learn
    elif selection == 4:
        F = get_full_rewards(maze)
        Q = learn_all(F)
        learned = True
    # Solve
    elif selection == 5:
        if not learned:
            print("Cannot do. AI must learn the maze")
        else:
            start, end = get_start_end()
            if not (start == "" or end == ""):
                end_cell_number = location_to_state[end]
                print(end_cell_number)
                q = Q[end_cell_number]
                route = get_route(q, start, end)
                if (len(route) == 0):
                    print("Stuck, can't print route")
                else:
                    print_route(maze, route)
    # Change dimensions
    elif selection == 8:
        width, height = get_maze_dimensions()
        if not (width == 0 or height == 0):
            maze_width = width
            maze_height = height
            initialize()
            learned = False
    elif selection == 9:
        processes = get_number_of_processes()
        if processes != 0:
            nprocesses = processes
