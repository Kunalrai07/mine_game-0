import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load data
with open("diamond_mines_dataset.json") as f:
    raw = json.load(f)

X = [x[0] for x in raw]
y = [x[1] for x in raw]

X = np.array(X)
y = np.array(y)

# Reshape for LSTM
X = X.reshape((X.shape[0], 1, 25))

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(1, 25), return_sequences=False))
model.add(Dense(64, activation='relu'))
model.add(Dense(25, activation='sigmoid'))  # One output per tile

model.compile(optimizer=Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# Save
model.save("diamond_lstm_model.h5")
print("Model saved as diamond_lstm_model.h5")
