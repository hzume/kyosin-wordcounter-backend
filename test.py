import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np

tag_list = ["名詞", "動詞", "形容詞", "副詞", "助動詞", "接続詞", "前置詞", "その他"]

with open("C:/Users/user/Desktop/test.txt", encoding="utf-8") as f:
    text = f.read()
morph = nltk.word_tokenize(text)
pos = nltk.pos_tag(morph)

lem = WordNetLemmatizer()
word_tag = []
for (w, tag) in pos:
    if tag in ["JJ", "JJR", "JJS"]:
        w = lem.lemmatize(w, pos="a")
        tag = "形容詞"
    elif tag in ["NN", "NNS", "NNP", "NNPS", "PRP", "PRP$"]:
        w = lem.lemmatize(w, pos="n")
        tag = "名詞"
    elif tag in ["RB", "RBR", "RBS"]:
        w = lem.lemmatize(w, pos="r")
        tag = "副詞"
    elif tag in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
        w = lem.lemmatize(w, pos="v")
        tag = "動詞"
    elif tag in ["IN"]:
        tag = "前置詞"
    elif tag in ["CC"]:
        tag = "接続詞"
    elif tag in ["MD"]:
        tag = "助動詞"
    else:
        tag = "その他"
    w = w.lower()
    word_tag.append((w, tag))
df = pd.DataFrame(word_tag, columns=["word", "tag"])
df = df.groupby(["word"])["tag"].value_counts()
df.name = "count"
df = pd.DataFrame(df).reset_index()
df = pd.pivot_table(df, index="word", columns="tag", values="count", aggfunc=np.sum)
df = df.fillna(0).astype("int").reset_index()

with open("C:/Users/user/Desktop/output.csv", encoding="utf-8") as f:
    output_df = pd.read_csv(f)

df = pd.concat([output_df, df]).fillna(0)
df = df.groupby("word")[tag_list].sum().reset_index()
df["count"] = df[tag_list].sum(axis=1)
df["品詞"] = df[tag_list].idxmax(axis=1)
df = df.sort_values("count", ascending=False)
df = df.reindex(columns=["word", "品詞", "count"] + tag_list)
print(df)
df.to_csv("C:/Users/user/Desktop/output.csv", index=False, encoding="utf_8_sig")