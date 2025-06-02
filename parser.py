import pymupdf
import pandas as pd
import warnings
import os
import logging
from PIL import Image
import pytesseract

warnings.filterwarnings("ignore")

global totalpages
totalpages=0

def resetpages():
    global totalpages
    totalpages=0

def parse_img(filepath, reset=False):
    global totalpages
    totalpages+=1
    text = pytesseract.image_to_string(Image.open(filepath), lang='eng')
    dct = {"page":[totalpages], "text":[text]}
    df = pd.DataFrame(dct)
    if os.path.isfile("texts.csv"):
        df_existing = pd.read_csv("texts.csv")
        df = pd.concat([df, df_existing], axis=0)
    df.to_csv("texts.csv", index=False)
    return text

def parse_pdf(filepath):
    pages = []
    texts = []
    book = pymupdf.open(filepath)
    global totalpages
    for i,page in enumerate(book):
        totalpages+=1
        text_tokenized = page.get_text().split()
        if not text_tokenized: 
            pix = page.get_pixmap()
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text_tokenized = pytesseract.image_to_string(image, lang='eng').split()
            if not text_tokenized: continue
        pages.append(totalpages)
        #pages.append(i+1)
        texts.append(" ".join(text_tokenized))#[:len(text_tokenized)//2]))
        #texts.append(" ".join(text_tokenized[len(text_tokenized)//2:]))
    dct = {"page":pages, "text":texts}
    df = pd.DataFrame(dct)
    if os.path.isfile("texts.csv"):
        df_existing = pd.read_csv("texts.csv")
        new_dct = {"page":pages, "text":texts}
        df = pd.DataFrame(new_dct)
        df = pd.concat([df_existing, df], axis=0)
    df.to_csv("texts.csv", index=False)

def main():
    book = pymupdf.open("uploads/APCalculusBCTextbook.pdf")
    print(parse_img("img2.png"))


if __name__=="__main__": main()