from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import nltk
from nltk.stem import WordNetLemmatizer
import pandas as pd

class Item(BaseModel):
    name: str
    text: str
    output: str

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
        output_df = pd.read_csv(item.output)
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
            else:
                tag = "その他"
            word_tag.append((w, tag))

        add_df = pd.DataFrame(word_tag, columns=["word", "tag"])
        add_df = add_df["word"].value_counts()
        add_df.name = "count"
        add_df.index.name = "word"
        add_df = pd.DataFrame(add_df).reset_index().groupby("word")["count"].sum().reset_index().sort_values("count", ascending=False)

        output_df = pd.concat([output_df, add_df]).groupby("word")["count"].sum().reset_index().sort_values("count", ascending=False)
        output_df.to_csv(item.output, index=False)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail="Can't open output file")
    except Exception as e:
        raise e