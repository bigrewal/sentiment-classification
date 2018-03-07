from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Embedding
from tensorflow.python.keras.optimizers import Adam

def sentiment_analysis(num_words, max_tokens):
    model = Sequential()

    #Embedding Layer. This layer will output the word vectors for each one of the words in the sentence
    embedding_size = 8
    model.add(Embedding(input_dim=num_words, 
                        output_dim=embedding_size,
                       input_length=max_tokens,
                       name='embedding_layer'))

    model.add(LSTM(units=16, return_sequences=True))
    model.add(LSTM(units=8, return_sequences=True))
    model.add(LSTM(units=4, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model
    
