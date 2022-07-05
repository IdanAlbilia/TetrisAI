import numpy as np


class TetrisAI:
    def __init__(self, grid, current_piece, next_piece, rows_cleared_with_last_move, model):
        self.grid = grid
        self.current_piece = current_piece
        self.next_piece = next_piece
        self.rows_cleared_with_last_move = rows_cleared_with_last_move
        self.model = model

    def generate_possible_moves(self, possible_moves):
        if len(possible_moves) == 0:
            return 0, None
        orig_grid = [row[:] for row in self.grid]
        best_move = [np.NINF, None]  # score, move
        for move in possible_moves:
            for i in range(len(move)):
                x_move, y_move = move[i]
                self.grid[y_move][x_move] = self.current_piece.color
            score = self.get_score()
            if score > best_move[0]:
                best_move[0], best_move[1] = score, move
            self.grid = [row[:] for row in orig_grid]
        self.grid = [row[:] for row in orig_grid]
        return best_move[0], best_move[1], orig_grid

    def get_board_info(self):
        peaks = self.get_peaks()
        agg_heights = self.get_aggregate_height(peaks)
        holes = self.get_holes(peaks)
        cols_with_holes = self.get_num_of_cols_with_holes(peaks)
        bumpiness = self.get_bumpiness(peaks)
        pits = self.get_num_of_pits(peaks)
        deepest_well = self.get_deepest_well(peaks)
        try:
            num_of_transitions_row = self.get_row_transition(max(peaks.values()))
        except:
            num_of_transitions_row = 0
        num_of_transitions_col = self.get_col_transition(peaks)
        cleared_lines = self.rows_cleared_with_last_move * 8

        return agg_heights, holes, bumpiness, cleared_lines, pits, deepest_well, cols_with_holes, \
               num_of_transitions_row, num_of_transitions_col

    def get_peaks(self):
        peaks = {}
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if self.grid[i][j] != (0, 0, 0):
                    if j not in peaks.keys():
                        peaks[j] = len(self.grid) - i
        return peaks

    def get_aggregate_height(self, peaks):
        return sum(peaks.values())

    def get_holes(self, peaks):
        holes = 0
        for key in peaks.keys():
            for row in range(peaks[key]):
                if self.grid[len(self.grid) - row - 1][key] == (0, 0, 0):
                    holes += 1
        return holes

    def get_num_of_cols_with_holes(self, peaks):
        cols = 0
        for key in peaks.keys():
            for row in range(peaks[key]):
                if self.grid[len(self.grid) - row - 1][key] == (0, 0, 0):
                    cols += 1
                    break
        return cols

    def get_bumpiness(self, peaks):
        bumps = 0
        for col in range(1, len(self.grid[0])):
            try:
                bumps += abs(peaks[col - 1] - peaks[col])
            except:
                try:
                    bumps += peaks[col - 1]
                except:
                    try:
                        bumps += peaks[col]
                    except:
                        pass
        return bumps

    def get_num_of_pits(self, peaks):
        return len(self.grid[0]) - len(peaks)

    def get_deepest_well(self, peaks):
        deepest_well = 0
        for col in range(1, len(self.grid[0])):
            try:
                deepest_well = max(deepest_well, abs(peaks[col - 1] - peaks[col]))
            except:
                try:
                    deepest_well = max(deepest_well, peaks[col - 1])
                except:
                    try:
                        deepest_well = max(deepest_well, peaks[col])
                    except:
                        pass
        return deepest_well

    def get_row_transition(self, highest_peak):
        sum = 0
        for row in range((len(self.grid) - highest_peak), len(self.grid)):
            for col in range(1, len(self.grid[0])):
                if self.grid[row][col] != self.grid[row][col - 1]:
                    sum += 1
        return sum

    def get_col_transition(self, peaks):
        sum = 0
        for col in range(len(self.grid[0])):
            try:
                if peaks[col] <= 1:
                    continue
                for row in range(len(self.grid) - peaks[col], len(self.grid) - 1):
                    if self.grid[row][col] != (0, 0, 0) and self.grid[row + 1][col] == (0, 0, 0):
                        sum += 1
                    elif self.grid[row][col] == (0, 0, 0) and self.grid[row + 1][col] != (0, 0, 0):
                        sum += 1
            except:
                pass
        return sum

    def get_score(self):
        inputs = self.get_board_info()
        output = self.model.activate(np.array(inputs))
        return output[0]
