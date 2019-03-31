# coding=utf-8

from tensorflow import keras
from sudoku import *

model = keras.models.Sequential([
	keras.layers.Dense(units=100, activation='relu', input_dim=81),
	keras.layers.Dense(units=100, activation='relu', input_dim=100),
	keras.layers.Dense(units=2, activation='softmax', input_dim=100)
])

s = Sudoku(10)
s.answer()
print(model.predict(s.data.reshape((1, -1))))
