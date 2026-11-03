"""
Self-contained Othello board state simulator and token vocabulary mapping
matching Li et al. (ICLR 2023) and Nanda, Lee & Wattenberg (BlackboxNLP 2023).
"""

import random
import numpy as np

ROWS = list("abcdefgh")
COLUMNS = [str(i) for i in range(1, 9)]
STARTING_SQUARES = [27, 28, 35, 36]
EIGHTS = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]

WHITE = -1
BLACK = 1
EMPTY = 0


def permit(s: str) -> int:
    s = s.lower()
    if len(s) != 2 or s[0] not in ROWS or s[1] not in COLUMNS:
        return -1
    return ROWS.index(s[0]) * 8 + COLUMNS.index(s[1])


def permit_reverse(integer: int) -> str:
    r, c = integer // 8, integer % 8
    return f"{ROWS[r]}{COLUMNS[c]}"


def build_itos() -> dict[int, int]:
    """Map model vocabulary index (0..60) to board cell index (0..63), skipping starting squares."""
    itos = {0: -100}
    for idx in range(1, 28):
        itos[idx] = idx - 1
    for idx in range(28, 34):
        itos[idx] = idx + 1
    for idx in range(34, 61):
        itos[idx] = idx + 3
    return itos


def build_stoi() -> dict[int, int]:
    """Map board cell index (0..63) to model vocabulary index (1..60)."""
    itos = build_itos()
    stoi = {v: k for k, v in itos.items()}
    stoi[-1] = 0
    for sq in STARTING_SQUARES:
        assert sq not in stoi
    return stoi


ITOS = build_itos()
STOI = build_stoi()


def to_board_label(cell_idx: int) -> str:
    return permit_reverse(cell_idx).upper()


class OthelloBoardState:
    """Exact 8x8 Othello rules simulator (1 = Black, -1 = White, 0 = Empty)."""

    def __init__(self, board_size: int = 8):
        self.board_size = board_size * board_size
        board = np.zeros((8, 8))
        board[3, 4] = BLACK
        board[3, 3] = WHITE
        board[4, 3] = BLACK
        board[4, 4] = WHITE
        self.initial_state = board
        self.state = self.initial_state.copy()
        self.age = np.zeros((8, 8))
        self.next_hand_color = BLACK
        self.history: list[int] = []

    def update(self, moves: list[int]):
        for move in moves:
            self.umpire(move)

    def umpire(self, move: int):
        r, c = move // 8, move % 8
        assert self.state[r, c] == 0, f"{r}-{c} is already occupied!"
        color = self.next_hand_color
        tbf = []
        for direction in EIGHTS:
            buffer = []
            cur_r, cur_c = r, c
            while True:
                cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                if cur_r < 0 or cur_r > 7 or cur_c < 0 or cur_c > 7:
                    break
                if self.state[cur_r, cur_c] == 0:
                    break
                elif self.state[cur_r, cur_c] == color:
                    tbf.extend(buffer)
                    break
                else:
                    buffer.append([cur_r, cur_c])
        if len(tbf) == 0:
            color *= -1
            self.next_hand_color *= -1
            for direction in EIGHTS:
                buffer = []
                cur_r, cur_c = r, c
                while True:
                    cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                    if cur_r < 0 or cur_r > 7 or cur_c < 0 or cur_c > 7:
                        break
                    if self.state[cur_r, cur_c] == 0:
                        break
                    elif self.state[cur_r, cur_c] == color:
                        tbf.extend(buffer)
                        break
                    else:
                        buffer.append([cur_r, cur_c])
        if len(tbf) == 0:
            valids = self.get_valid_moves()
            if len(valids) == 0:
                raise RuntimeError("Both colors cannot put piece, game should have ended!")
            else:
                raise RuntimeError("Illegal move!")

        self.age += 1
        for ff in tbf:
            self.state[ff[0], ff[1]] *= -1
            self.age[ff[0], ff[1]] = 0
        self.state[r, c] = color
        self.age[r, c] = 0
        self.next_hand_color *= -1
        self.history.append(move)

    def tentative_move(self, move: int) -> int:
        r, c = move // 8, move % 8
        if self.state[r, c] != 0:
            return 0
        color = self.next_hand_color
        tbf = []
        for direction in EIGHTS:
            buffer = []
            cur_r, cur_c = r, c
            while True:
                cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                if cur_r < 0 or cur_r > 7 or cur_c < 0 or cur_c > 7:
                    break
                if self.state[cur_r, cur_c] == 0:
                    break
                elif self.state[cur_r, cur_c] == color:
                    tbf.extend(buffer)
                    break
                else:
                    buffer.append([cur_r, cur_c])
        if len(tbf) != 0:
            return 1
        color *= -1
        for direction in EIGHTS:
            buffer = []
            cur_r, cur_c = r, c
            while True:
                cur_r, cur_c = cur_r + direction[0], cur_c + direction[1]
                if cur_r < 0 or cur_r > 7 or cur_c < 0 or cur_c > 7:
                    break
                if self.state[cur_r, cur_c] == 0:
                    break
                elif self.state[cur_r, cur_c] == color:
                    tbf.extend(buffer)
                    break
                else:
                    buffer.append([cur_r, cur_c])
        if len(tbf) == 0:
            return 0
        return 2

    def get_valid_moves(self) -> list[int]:
        regular_moves = []
        forfeit_moves = []
        for move in range(64):
            x = self.tentative_move(move)
            if x == 1:
                regular_moves.append(move)
            elif x == 2:
                forfeit_moves.append(move)
        if regular_moves:
            return regular_moves
        elif forfeit_moves:
            return forfeit_moves
        return []


def generate_synthetic_game(seed: int | None = None) -> list[int]:
    if seed is not None:
        random.seed(seed)
    moves = []
    board = OthelloBoardState()
    valid = board.get_valid_moves()
    while valid:
        m = random.choice(valid)
        moves.append(m)
        board.update([m])
        valid = board.get_valid_moves()
    return moves
