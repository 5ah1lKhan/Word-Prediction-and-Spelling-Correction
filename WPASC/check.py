import pandas as pd
df = pd.read_csv("fake_or_real_news.csv")
text = " ".join(list(df.text.values))
with open("data/joined_text.txt", "w", encoding="utf-8") as f:
    f.write(text)