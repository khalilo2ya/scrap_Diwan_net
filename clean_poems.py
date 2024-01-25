import pandas as pd
import re

# Load the data from Excel
file_path = 'poems.xlsx'
df = pd.read_excel(file_path)

# Define the cleaning function
def clean_arabic_poetry(poem):
    # Remove punctuation (keeping Arabic punctuation marks)
    poem = re.sub(r'[^\w\sء-ي]', '', poem)
    
    # Remove extra spaces
    poem = re.sub(r'\s+', ' ', poem).strip()
    
    # Normalize characters (optional)
    # You can add more normalization rules as needed
    poem = re.sub(r'[إأآ]', 'ا', poem)
    
    # Remove numbers
    poem = re.sub(r'\d', '', poem)
    
    return poem

# Apply the cleaning function to the 'poem_content' column
df['cleaned_poem_content'] = df['poem_content'].apply(clean_arabic_poetry)

# Save the cleaned data to a new Excel file
output_file_path = 'cleaned_poems.xlsx'
df.to_excel(output_file_path, index=False)

print(f'Cleaned data saved to {output_file_path}')
