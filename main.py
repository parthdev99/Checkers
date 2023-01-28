start_board = [1, 1, 1, 1,  1, 1, 1, 0,  1, 0, 0, 1,  0, 0, 1, 0,  0, 0, 0, 0,  0, 0, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1]
start_board = checkers.expand(start_board)
next_board = checkers.expand(best_move(start_board))

print("Starting position : ")
print_board(start_board)

print("\nBest next move : ")
print_board(next_board)