import checkers
import metrics_model
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.models import model_from_json

# Board model
board_model = Sequential()

# input dimensions is 32 board position values
board_model.add(Dense(64 , activation='relu', input_dim=32))

# use regularizers, to prevent fitting noisy labels
board_model.add(Dense(32 , activation='relu', kernel_regularizer=regularizers.l2(0.01)))
board_model.add(Dense(16 , activation='relu', kernel_regularizer=regularizers.l2(0.01))) # 16
board_model.add(Dense(8 , activation='relu', kernel_regularizer=regularizers.l2(0.01))) # 8

# output isn't squashed, because it might lose information
board_model.add(Dense(1 , activation='linear', kernel_regularizer=regularizers.l2(0.01)))
board_model.compile(optimizer='nadam', loss='binary_crossentropy')

# calculate heuristic metric for data
metrics = np.zeros((0, 10))
winning = np.zeros((0, 1))
data = metrics_model.boards_list

for board in data:
	temp = checkers.get_my_metrics(board)
	metrics = np.vstack((metrics, temp[1:]))
	winning = np.zeros((0, 1))
 
# calculate probilistic (noisy) labels
probabilistic = metrics_model.metrics_model.predict_on_batch(metrics)

# fit labels to {-1, 1}
probabilistic = np.sign(probabilistic)

# calculate confidence score for each probabilistic label using error between probabilistic and weak label
confidence = 1/(1 + np.absolute(winning - probabilistic[:, 0]))
 
# pass to the Board model
board_model.fit(data, probabilistic, epochs=32, batch_size=64, sample_weight=confidence, verbose=0)

board_json = board_model.to_json()
with open('board_model.json', 'w') as json_file:
	json_file.write(board_json)
board_model.save_weights('board_model.h5')

print('Checkers Board Model saved to: board_model.json/h5')