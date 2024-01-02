import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch.utils.data import Dataset, DataLoader

# Read poems from Excel file
file_path = 'poems.xlsx'
df = pd.read_excel(file_path)
poems = df['poem_content'].tolist()

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token
print("Tokenizer vocabulary size:", len(tokenizer))

# Tokenize poems
tokenized_poems = tokenizer(poems, return_tensors='pt', padding=True, truncation=True)
print("Max token length after tokenization:", tokenized_poems['input_ids'].shape[1])

# Create a custom dataset class
class PoemsDataset(Dataset):
    def __init__(self, tokenized_poems):
        self.input_ids = tokenized_poems['input_ids']
        self.attention_mask = tokenized_poems['attention_mask']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }

# Create an instance of the custom dataset
dataset = PoemsDataset(tokenized_poems)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for epoch in range(5):  # Choose the number of epochs
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Mask padding tokens

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')

# Generate text using the fine-tuned model
generated_poem = tokenizer.generate(input_ids=tokenizer.encode("ناهضٌ من بذوري إليهما"), max_length=200, do_sample=True)
generated_text = tokenizer.decode(generated_poem[0], skip_special_tokens=True)
print(generated_text)
