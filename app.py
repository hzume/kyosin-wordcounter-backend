from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np

class Item(BaseModel):
    name: str
    text: str
    output: str

tag_list = ["名詞", "動詞", "形容詞", "副詞", "助動詞", "接続詞", "前置詞", "その他"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return "This is Kyosin-WordCounter's backend api server."

@app.post("/read_item/")
def read_item(item: Item):
    try:
        output_df = pd.read_csv(item.output, encoding="utf_8_sig")
    except FileNotFoundError as e:
        raise HTTPException(status_code=406, detail="Output file not found")

    try:
        morph = nltk.word_tokenize(item.text)
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

        output_df = pd.concat([output_df, df]).fillna(0)
        output_df = output_df.groupby("word")[tag_list].sum().reset_index()
        output_df["count"] = output_df[tag_list].sum(axis=1)
        output_df["品詞"] = output_df[tag_list].idxmax(axis=1)
        output_df = output_df.sort_values("count", ascending=False).reset_index(drop=True)
        output_df = output_df.reindex(columns=["word", "品詞", "count"] + tag_list)
        output_df.to_csv(item.output, encoding="utf_8_sig", index=False)

    except PermissionError as e:
        raise HTTPException(status_code=403, detail="Can't open output file")
    except Exception as e:
        raise e