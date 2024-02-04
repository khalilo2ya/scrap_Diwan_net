import numpy as np
import pandas as pd
import json
import tkinter as tk
from tkinter import ttk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

def generate_poem(seed_text, next_words, model, tokenizer, max_sequence_len):
    poem = seed_text
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
        poem += " " + output_word

    return poem

def on_generate_button():
    seed_text = entry_seed.get()
    generated_poem = generate_poem(seed_text, 100, model, tokenizer, max_sequence_len)
    text_generated.delete("1.0", tk.END)
    text_generated.insert(tk.END, generated_poem)

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

# Load pre-trained model
model = load_model('arabic_poetry_generator_.h5')

# GUI
root = tk.Tk()
root.title("Arabic Poetry Generator")

# Entry for seed text
label_seed = ttk.Label(root, text="Seed Text:")
label_seed.pack()
entry_seed = ttk.Entry(root)
entry_seed.pack()

# Button to generate poem
generate_button = ttk.Button(root, text="Generate Poem", command=on_generate_button)
generate_button.pack()

# Text widget to display generated poem
text_generated = tk.Text(root, wrap=tk.WORD, height=10, width=40)
text_generated.pack()

root.mainloop()
