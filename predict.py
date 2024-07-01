import numpy as np
import pandas as pd
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load data from JSON
with open('poems_lite.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

df = pd.DataFrame(data)

# Combine text fields
df['combined'] = df['poet_name'] + ' ' + df['poem_name'] + ' ' + df['poem_content']

# Preprocessing: Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['combined'])
total_words = len(tokenizer.word_index) + 1

# Padding sequences
max_sequence_len = 306  # Replace with the appropriate sequence length used during training
# max_sequence_len = max(len(sequence.split()) for sequence in df['combined'])
print("Max Sequence Length:", max_sequence_len)
# Load pre-trained model
model = load_model('arabic_poetry_generator_.h5')



seed_text = "حبيبي"
next_words = 100

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
    predicted_probs = model.predict(token_list)[0]
    
    # Sample a word index from the predicted probabilities
    predicted_index = np.random.choice(len(predicted_probs), p=predicted_probs)
    
    # Convert index to word
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            output_word = word
            break
            
    seed_text += " " + output_word
print(seed_text)
