import pandas as pd  # Importing pandas for data handling
import numpy as np  # Importing numpy for numerical operations
from tensorflow.keras.preprocessing.text import Tokenizer  # Importing Tokenizer for text tokenization
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Importing pad_sequences for padding sequences
from tensorflow.keras.models import Sequential  # Importing Sequential model from Keras
from tensorflow.keras.layers import Embedding, LSTM, Dense  # Importing specific layers from Keras

# Load data from Excel
file_path = 'poems_content_lite.xlsx'  # Path to the Excel file containing poems
df = pd.read_excel(file_path)  # Reading the Excel file into a pandas DataFrame

# Combine poet_name, poem_name, poem_content into a single text
# df['combined'] = df['poet_name'] + ' ' + df['poem_name'] + ' ' + df['poem_content']  # Combining multiple columns into one
df['combined'] = df['poem_content']  # Combining multiple columns into one

# Preprocessing: Tokenization
tokenizer = Tokenizer()  # Initializing Tokenizer
tokenizer.fit_on_texts(df['combined'])  # Fitting tokenizer on the combined text
total_words = len(tokenizer.word_index) + 1  # Total number of unique words after tokenization

input_sequences = []  # List to store input sequences
for line in df['combined']:  # Loop through each line in the combined text
    token_list = tokenizer.texts_to_sequences([line])[0]  # Tokenizing each line
    for i in range(1, len(token_list)):  # Loop through tokens in the token list
        n_gram_sequence = token_list[:i+1]  # Creating n-gram sequences
        input_sequences.append(n_gram_sequence)  # Appending sequences to input_sequences list

# Padding sequences
max_sequence_len = max([len(x) for x in input_sequences])  # Finding the maximum sequence length
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len+1, padding='pre'))  # Padding sequences and converting to numpy array

# Creating predictors and label
predictors, label = input_sequences[:, :-1], input_sequences[:, -1]  # Splitting predictors and labels

# Model creation
model = Sequential()  # Initializing a Sequential model
model.add(Embedding(total_words, 100, input_length=max_sequence_len))  # Adding Embedding layer
model.add(LSTM(150, return_sequences=True))  # Adding LSTM layer with return sequences
model.add(LSTM(150))  # Adding another LSTM layer
model.add(Dense(total_words, activation='softmax'))  # Adding Dense layer with softmax activation

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # Compiling the model

# Define a generator function for sequences
def generate_sequences(input_sequences, label, batch_size):  # Defining a generator function
    while True:  # Infinite loop
        for i in range(0, len(input_sequences), batch_size):  # Loop through sequences in batch_size
            yield input_sequences[i:i+batch_size], label[i:i+batch_size]  # Yielding batches of sequences and labels

# Initialize the generator
batch_size = 32  # Setting batch size
sequence_generator = generate_sequences(predictors, label, batch_size=batch_size)  # Creating sequence generator

# Train the model using the generator
model.fit(sequence_generator, steps_per_epoch=len(predictors)//batch_size, epochs=100)  # Training the model

model.save('arabic_poetry_generator.h5')  # Saving the trained model

# Generating new poetry
seed_text = "اذا الشعب"  # Initial seed text for generating poetry
next_words = 100  # Number of words to generate

for _ in range(next_words):  # Loop to generate next_words
    token_list = tokenizer.texts_to_sequences([seed_text])[0]  # Tokenizing the seed text
    token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')  # Padding the token list
    predicted = np.argmax(model.predict(token_list), axis=-1)  # Predicting the next word index
    output_word = ""  # Initializing the output word
    for word, index in tokenizer.word_index.items():  # Loop through word_index
        if index == predicted:  # Finding the word corresponding to the predicted index
            output_word = word  # Assigning the word to output_word
            break  # Break the loop when word is found
    seed_text += " " + output_word  # Appending the predicted word to seed_text

print(seed_text)  # Printing the generated poetry
