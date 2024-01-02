import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

# Read poems from Excel file
file_path = 'poems.xlsx'
df = pd.read_excel(file_path)
poems = df['poem_content'].tolist()

# Initialize the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(poems)
total_words = len(tokenizer.word_index) + 1

# Create input sequences and output sequences
input_sequences = []
output_sequences = []

for poem in poems:
    token_list = tokenizer.texts_to_sequences([poem])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence[:-1])
        output_sequences.append(n_gram_sequence[-1])

# Pad input sequences
max_sequence_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Convert output sequences to one-hot encoded vectors
output_sequences = np.array(output_sequences)
output_sequences = np.eye(total_words)[output_sequences]

# Define the LSTM model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_length-1))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(input_sequences, output_sequences, epochs=100, batch_size=64)

# Function to generate text
def generate_text(seed_text, next_words=50):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)[0]
        predicted_word_index = np.random.choice(total_words, 1, p=predicted_probs)[0]
        predicted_word = tokenizer.index_word[predicted_word_index]
        seed_text += " " + predicted_word
    return seed_text

# Example usage
seed_text = "ناهضٌ من بذوري إليهما"
generated_poem = generate_text(seed_text, next_words=50)
print(generated_poem)
