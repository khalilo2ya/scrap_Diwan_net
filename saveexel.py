import pandas as pd
import json
with open('poems.json') as f:
    poems = json.load(f)

df = pd.DataFrame(poems)

df.to_excel('poems.xlsx', index=False)