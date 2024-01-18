import pandas as pd
import urllib3
from farasa.pos import FarasaPOSTagger

# Disable warnings about an unverified HTTPS request
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load the Excel file
file_path = 'poems.xlsx'
df = pd.read_excel(file_path)

# Function to remove diacritics from Arabic text
def remove_diacritics(text):
    farasa_pos = FarasaPOSTagger()
    tagged_text = farasa_pos.tag(text)
    diacritic_removed_text = ' '.join([word.split('/')[0] for word in tagged_text.split()])
    return diacritic_removed_text

# Apply the function to the "poem_content" column
df['poem_content'] = df['poem_content'].apply(remove_diacritics)

# Save the modified DataFrame to a new Excel file
output_file_path = 'poems_without_diacritics.xlsx'
df.to_excel(output_file_path, index=False)

print(f'DataFrame with removed diacritics saved to {output_file_path}')
