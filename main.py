import checkers
import numpy as np
import reinforcement_of_model

def best_move(board):
  compressed_board = checkers.compress(board)
  boards = np.zeros((0, 32))
  boards = checkers.generate_next(board)
  scores = reinforcement_of_model.reinforced_model.predict_on_batch(boards)
  max_index = np.argmax(scores)
  best = boards[max_index]
  return best

def print_board(board):
  for row in board:
    for square in row:
      if square == 1:
        caracter = "|O"
      elif square == -1:
        caracter = "|X"
      else:
        caracter = "| "
      print(str(caracter), end='')
    print('|')


start_board = [1, 1, 1, 1,  1, 1, 1, 0,  1, 0, 0, 1,  0, 0, 1, 0,  0, 0, 0, 0,  0, 0, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1]
start_board = checkers.expand(start_board)
next_board = checkers.expand(best_move(start_board))

print("Starting position : ")
print_board(start_board)

print("\nBest next move : ")
print_board(next_board)