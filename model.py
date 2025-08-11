from keras._tf_keras.keras.datasets import imdb
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.callbacks import EarlyStopping

num_words = 20000  
maxlen = 250        
embedding_dim = 100 

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

model = Sequential([

    Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=maxlen),


    Bidirectional(LSTM(64, return_sequences=False)),

    Dropout(0.5),

    Dense(64, activation='relu'),

    Dropout(0.5),

    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(x_train, y_train,
          epochs=10,
          batch_size=128,  
          validation_data=(x_test, y_test),
          callbacks=[early_stop],
          verbose=1)

model.save('modele_imdb_bilstm.h5')

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Précision sur le test : {accuracy:.2f}")
