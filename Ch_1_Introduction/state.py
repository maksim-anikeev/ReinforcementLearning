import numpy as np

TOKEN_MAP = {
    1: 'x',
    0: ' ',
    -1: 'o'
}

class State:
    def __init__(self, board_size: int = 3):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size))
        self.hash_val = None
        self._is_end = None

    @property
    def hash(self):
        if self.hash_val is None:
            self.hash_val = 0
            for i in np.nditer(self.board):
                self.hash_val = int(self.hash_val * 3 + i + 1)
        return self.hash_val
    
    @property
    def is_end(self):
        if self._is_end is not None:
            return self._is_end
        
        if self._check_lines(rows=True):
            self._is_end = True
            return self._is_end
        if self._check_lines(rows=False):
            self._is_end = True
            return self._is_end
        if self._check_diagonals():
            self._is_end = True
            return self._is_end
        if self._check_tie():
            self._is_end = True
            return self._is_end
        self._is_end = False
        return self._is_end
            
    def _check_lines(self, rows=True):
        for i in range(self.board_size):
            line = self.board[i, :] if rows else self.board[:, i]
            line_value = np.sum(line)
            if np.abs(line_value) == 3:
                self.winner = line_value / 3
                return True
        return False
    
    def _check_diagonals(self):
        d1_value = 0
        d2_value = 0
        for i in range(self.board_size):
            d1_value += self.board[i, i]
            d2_value += self.board[i, self.board_size - i - 1]
        if d1_value == 3 or d2_value == 3:
            self.winner = 1
            return True
        if d1_value == -3 or d2_value == 3:
            self.winner = -1
            return True
        return False
    
    def _check_tie(self):
        sum_value = np.sum(np.abs(self.board))
        if sum_value == self.board_size ** 2:
            self.winner = 0
            return True
        return False

    def print_state(self):
        for i in range(self.board_size):
            print('-------------')
            output_line = '| '
            for j in range(self.board_size):
                output_line += TOKEN_MAP[self.board[i, j]] + ' | '
            print(output_line)
        print('-------------')
    
    def next_state(self, i: int, j: int, symbol):
        new_state = State()
        new_state.board = np.copy(self.board)
        new_state.board[i, j] = symbol
        return new_state