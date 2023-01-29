import checkers
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.models import model_from_json

# Metrics model, which only looks at heuristic scoring metrics used for labeling
metrics_model = Sequential()
metrics_model.add(Dense(32, activation='relu', input_dim=10)) 
metrics_model.add(Dense(16, activation='relu',  kernel_regularizer=regularizers.l2(0.1)))

# output is passed to relu() because labels are binary
metrics_model.add(Dense(1, activation='relu',  kernel_regularizer=regularizers.l2(0.1)))
metrics_model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=["acc"])

start_board = checkers.expand(checkers.np_board())
boards_list = checkers.generate_next(start_board)
branching_position = 0
nmbr_generated_game = 1000
while len(boards_list) < nmbr_generated_game:
	temp = len(boards_list) - 1
	for i in range(branching_position, len(boards_list)):
		if (checkers.possible_moves(checkers.reverse(checkers.expand(boards_list[i]))) > 0):
				boards_list = np.vstack((boards_list, checkers.generate_next(checkers.reverse(checkers.expand(boards_list[i])))))
	branching_position = temp

# calculate/save heuristic metrics for each game state
metrics	= np.zeros((0, 10))
winning = np.zeros((0, 1))

for board in boards_list[:nmbr_generated_game]:
	temp = checkers.get_my_metrics(board)
	metrics = np.vstack((metrics, temp[1:]))
	winning  = np.vstack((winning, temp[0]))
 
# fit the metrics model
history = metrics_model.fit(metrics , winning, epochs=32, batch_size=64, verbose=0)


# # for plotting
# # History for accuracy
# plot.plot(history.history['acc'])
# plot.plot(history.history['val_acc'])
# plot.title('model accuracy')
# plot.ylabel('accuracy')
# plot.xlabel('epoch')
# plot.legend(['train', 'validation'], loc='upper left')
# plot.show()

# # History for loss
# plot.plot(history.history['loss'])
# plot.plot(history.history['val_loss'])
# plot.title('model loss')
# plot.ylabel('loss')
# plot.xlabel('epoch')
# plot.legend(['train', 'validation'], loc='upper left')
# plot.show()