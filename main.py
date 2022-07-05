import time
from multiprocessing import Pool, freeze_support

import numpy as np
import pygame
import random
import AI

import io
import neat
import os
import pickle
import visualize

# creating the data structure for pieces
# setting up global vars
# functions
# - create_grid
# - draw_grid
# - draw_window
# - rotating shape in main
# - setting up the main
from Network import Network

"""
10 x 20 square grid
shapes: S, Z, I, O, J, L, T
represented in order by 0 - 6
"""

pygame.font.init()

# GLOBALS VARS
s_width = 800
s_height = 700
play_width = 300  # meaning 300 // 10 = 30 width per block
play_height = 600  # meaning 600 // 20 = 20 height per block
block_size = 30

top_left_x = (s_width - play_width) // 2
top_left_y = s_height - play_height

# AI VARS
epochs = 20
max_fitness = 0
run_per_child = 3
max_score = 999999
blank_tile = 47
pop_size = 50
maximum_score = 999999
n_workers = 5

# SHAPE FORMATS

S = [['.....',
      '......',
      '..00..',
      '.00...',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['..0..',
      '..0..',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]

L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]

shapes = [S, Z, I, O, J, L, T]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]
shape_names = {(0, 255, 0): 'S', (255, 0, 0): 'Z', (0, 255, 255): 'I', (255, 255, 0): 'O', (255, 165, 0): 'J',
               (0, 0, 255): 'L', (128, 0, 128): 'T'}


# index 0 - 6 represent shape


class Piece(object):
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = 0
        self.shape_name = shape_names[self.color]


def update_scores(nscore):
    score = get_max_score()
    with open('scores.txt', 'w') as f:
        if int(score) > nscore:
            f.write(str(score))
        else:
            f.write(str(nscore))


def get_max_score():
    with open('scores.txt', 'r') as f:
        lines = f.readlines()
        score = lines[0].strip()
    return score


def create_grid(locked_positions={}):
    grid = [[(0, 0, 0) for _ in range(10)] for _ in range(20)]

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if (j, i) in locked_positions:
                c = locked_positions[(j, i)]
                grid[i][j] = c
    return grid


def convert_shape_format(shape):
    positions = []
    format = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                positions.append((shape.x + j, shape.y + i))

    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)

    return positions


def valid_space(shape, grid):
    accepted_pos = [[(j, i) for j in range(10) if grid[i][j] == (0, 0, 0)] for i in range(20)]
    accepted_pos = [j for sub in accepted_pos for j in sub]

    formatted = convert_shape_format(shape)

    for pos in formatted:
        if pos not in accepted_pos:
            if pos[1] > -1:
                return False
    return True


def valid_space_pos(shape, grid):
    accepted_pos = [[(j, i) for j in range(10) if grid[i][j] == (0, 0, 0)] for i in range(20)]
    accepted_pos = [j for sub in accepted_pos for j in sub]

    for pos in shape:
        if tuple(pos) not in accepted_pos:
            if pos[1] > -1:
                return False
    return True


def check_lost(positions, possible_moves):
    if len(possible_moves) == 0:
        return True
    for pos in positions:
        x, y = pos
        if y < 1:
            return True

    return False


def get_shape():
    return Piece(5, 0, random.choice(shapes))


def draw_text_middle(text, size, color, surface):
    font = pygame.font.SysFont('comicsans', size, bold=True)
    label = font.render(text, 1, color)

    surface.blit(label, (
        top_left_x + play_width / 2 - (label.get_width() / 2), top_left_y + play_height / 2 - label.get_height()))


def draw_grid(surface, grid):
    sx = top_left_x
    sy = top_left_y

    for i in range(len(grid)):
        pygame.draw.line(surface, (128, 128, 128), (sx, sy + i * block_size), (sx + play_width, sy + i * block_size))
        for j in range(len(grid[i])):
            pygame.draw.line(surface, (128, 128, 128), (sx + j * block_size, sy),
                             (sx + j * block_size, sy + play_height))


def clear_rows(grid, locked):
    inc = 0
    for i in range(len(grid) - 1, -1, -1):
        row = grid[i]
        if (0, 0, 0) not in row:
            inc += 1
            ind = i
            for j in range(len(row)):
                try:
                    del locked[(j, i)]
                except:
                    continue
    if inc > 0:
        for key in sorted(list(locked), key=lambda x: x[1])[::-1]:
            x, y = key
            if y < ind:
                new_key = (x, y + inc)
                locked[new_key] = locked.pop(key)
    return inc, locked


def draw_next_shape(shape, surface):
    font = pygame.font.SysFont('comicsans', 30)
    label = font.render('Next Shape', 1, (255, 255, 255))

    sx = top_left_x + play_width + 50
    sy = top_left_y + play_height / 2 - 100

    format = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                pygame.draw.rect(surface, shape.color,
                                 (sx + j * block_size, sy + i * block_size, block_size, block_size), 0)
    surface.blit(label, (sx + 7, sy - 50))


def draw_window(surface, grid, score=0, best_score='0'):
    surface.fill((0, 0, 0))

    pygame.font.init()
    font = pygame.font.SysFont('comicsans', 60)
    label = font.render('Tetris', 1, (255, 255, 255))

    surface.blit(label, (top_left_x + play_width / 2 - (label.get_width() / 2), 20))

    # current score
    font = pygame.font.SysFont('comicsans', 30)
    label = font.render('Score: ' + str(score), 1, (255, 255, 255))

    sx = top_left_x + play_width + 50
    sy = top_left_y + play_height / 2 - 100

    surface.blit(label, (sx + 20, sy + 160))

    # high_score
    font = pygame.font.SysFont('comicsans', 30)
    label = font.render('High Score: ' + str(best_score), 1, (255, 255, 255))

    sx = top_left_x - 250
    sy = top_left_y + 200

    surface.blit(label, (sx + 20, sy + 160))

    for i in range(len(grid)):
        for j in range(len(grid[i])):
            pygame.draw.rect(surface, grid[i][j],
                             (top_left_x + j * block_size, top_left_y + i * block_size, block_size, block_size), 0)

    pygame.draw.rect(surface, (255, 0, 0), (top_left_x, top_left_y, play_width, play_height), 4)

    draw_grid(surface, grid)


# def main(win, run_per_child):
def main(genome, config):
    win = pygame.display.set_mode((s_width, s_height))
    pygame.display.set_caption('Tetris')
    win.fill((0, 0, 0))
    model = neat.nn.FeedForwardNetwork.create(genome, config)
    child_fitness = 0

    run_number = 0

    best_score = get_max_score()
    locked_positions = {}

    change_piece = False
    run = True
    current_piece = get_shape()
    next_piece = get_shape()
    clock = pygame.time.Clock()
    fall_time = 0
    fall_speed = 0.27
    level_time = 0
    score = 0

    scores = []

    rows_cleared = 0

    while run and run_number < run_per_child:
        grid = create_grid(locked_positions)
        fall_time += clock.get_rawtime()
        level_time += clock.get_rawtime()
        clock.tick()

        if level_time / 1000 > 5:
            level_time = 0
            if fall_speed > 0.12:
                fall_speed -= 0.005

        if fall_time / 1000 > fall_speed:
            fall_time = 0
            current_piece.y += 1
            if not (valid_space(current_piece, grid)) and current_piece.y > 0:
                current_piece.y -= 1

        ai = AI.TetrisAI(grid, current_piece, next_piece, rows_cleared, model)
        possible_moves = generate_possible_moves(current_piece, grid)
        if len(possible_moves) != 0:
            move_score, shape_pos, grid = ai.generate_possible_moves(possible_moves)

        change_piece = True

        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         run = False
        #         pygame.display.quit()
        #
        #     # User is playing..
        #     if event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_LEFT:
        #             current_piece.x -= 1
        #             if not (valid_space(current_piece, grid)):
        #                 current_piece.x += 1
        #         if event.key == pygame.K_RIGHT:
        #             current_piece.x += 1
        #             if not (valid_space(current_piece, grid)):
        #                 current_piece.x -= 1
        #         if event.key == pygame.K_DOWN:
        #             current_piece.y += 1
        #             if not (valid_space(current_piece, grid)):
        #                 current_piece.y -= 1
        #         if event.key == pygame.K_UP:
        #             current_piece.rotation += 1
        #             if not (valid_space(current_piece, grid)):
        #                 current_piece.rotation -= 1

        # shape_pos = convert_shape_format(current_piece)

        if shape_pos is not None and len(possible_moves) != 0:
            for i in range(len(shape_pos)):
                x, y = shape_pos[i]
                if y > -1:
                    grid[y][x] = current_piece.color

            if change_piece:
                for pos in shape_pos:
                    p = (pos[0], pos[1])
                    if p in locked_positions.keys():
                        break
                    else:
                        locked_positions[p] = current_piece.color
                current_piece = next_piece
                next_piece = get_shape()

                rows_cleared, locked_positions = clear_rows(grid, locked_positions)

                check_clear_rows(grid, locked_positions)
                grid = create_grid(locked_positions)

                score += rows_cleared * 10

        draw_window(win, grid, score, best_score)
        draw_next_shape(next_piece, win)
        pygame.display.update()

        if check_lost(locked_positions, possible_moves):
            # draw_text_middle("You Lost!", 80, (255, 255, 255), win)
            # run = False
            update_scores(score)
            run_number += 1
            scores.append(score)
            score = 0

    child_fitness = np.average(scores)
    return child_fitness


def check_clear_rows(grid, locked):
    empty_row_index = np.Inf
    full_row_first_index = np.Inf
    for i in range(len(grid) - 1, -1, -1):
        row = grid[i]
        if all([key == (0,0,0) for key in row]) and empty_row_index == np.Inf:
            empty_row_index = i
        if any([key != (0, 0, 0) for key in row]) and empty_row_index != np.Inf:
            full_row_first_index = i
    if empty_row_index > full_row_first_index:
        for key in sorted(list(locked), key=lambda x: x[1])[::-1]:
            x, y = key
            if y < empty_row_index:
                new_key = (x, y + 1)
                locked[new_key] = locked.pop(key)



def generate_possible_moves(current_piece, grid):
    move_end_positions = []
    orig_x, orig_y = current_piece.x, current_piece.y
    flips = len(current_piece.shape)
    for j in range(flips):
        current_piece.rotation = j
        for i in range(len(grid[0])):
            current_piece.x = i
            current_piece.y = orig_y
            while valid_space(current_piece, grid):
                current_piece.y += 1
                if (not (valid_space(current_piece, grid))) and current_piece.y > 0:
                    current_piece.y -= 1
                    shape_pos = convert_shape_format(current_piece)
                    if check_valid_pos(shape_pos, grid):
                        move_end_positions.append(shape_pos)
                    break
    current_piece.x, current_piece.y = orig_x, orig_y
    return move_end_positions


def check_valid_pos(pos, grid):
    for i in range(len(pos)):
        x, y = pos[i]
        if x < 0 or y < 0 or x >= len(grid[0]) or y >= len(grid):
            return False
    return True


# def main_menu(win):
#     run = True
#     while run:
#         win.fill((0, 0, 0))
#         draw_text_middle('Press Any Key To Play', 60, (255, 255, 255), win)
#         pygame.display.update()
#         # for event in pygame.event.get():
#         #     if event.type == pygame.QUIT:
#         #         run = False
#         #     if event.type == pygame.KEYDOWN:
#         # main(win, run_per_child)
#     pygame.display.quit()


local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config', 'config-feedforward.txt')


def run(config):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    p = neat.Population(config)

    # p = neat.Checkpointer().restore_checkpoint('checkpoint/neat-checkpoint-2')
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(
        neat.Checkpointer(1, filename_prefix='checkpoint/neat-checkpoint-'))

    pe = neat.ParallelEvaluator(n_workers, main)
    winner = p.run(pe.evaluate, epochs)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    node_names = {-1: 'agg_height', -2: 'n_holes', -3: 'bumpiness',
                  -4: 'cleared', -5: 'num_pits', -6: 'max_wells',
                  -7: 'n_cols_with_holes', -8: 'row_transitions',
                  -9: 'col_transitions', 0: 'Score'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_feedforward.txt')
    run(config_path)
    # win = pygame.display.set_mode((s_width, s_height))
    # pygame.display.set_caption('Tetris')
    # main_menu(win)  # start game
