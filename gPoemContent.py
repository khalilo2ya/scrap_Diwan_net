import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load data from Excel
file_path = 'poems_content.xlsx'
df = pd.read_excel(file_path)

# Combine poet_name, poem_name, poem_content into a single text
df['combined'] = df['poem_content']

# Preprocessing: Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['combined'])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in df['combined']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Padding sequences
max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len+1, padding='pre')


# Creating predictors and label
input_sequences = input_sequences[:, :-1]
predictors, label = input_sequences[:, :-1], input_sequences[:, -1]

# Model creation
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))
model.add(LSTM(150, return_sequences=True))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(predictors, label, epochs=100, verbose=1)

model.save('arabic_poetry_generator_.h5')

# Generating new poetry
seed_text = "اذا الشعب"
next_words = 100

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')  # Adjusted to use max_sequence_len
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word

print(seed_text)
