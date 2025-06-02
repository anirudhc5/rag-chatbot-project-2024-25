from deepgram import DeepgramClient, PrerecordedOptions
import deepgram
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
import asyncio
import httpx
import pandas as pd

load_dotenv()

def parse_audio(filepath):
    deepgram = DeepgramClient("0e923c39d37107660f3483d9e74ec7013310de42")
    response = ""
    with open(filepath, 'rb') as buffer_data:
        payload = { 'buffer': buffer_data }

        options = PrerecordedOptions(
            smart_format=True, model="nova-2", language="en-US"
        )
        response = deepgram.listen.prerecorded.v('1').transcribe_file(payload, options)
    paragraphs = response["results"]["channels"][0]["alternatives"][0]["paragraphs"]["paragraphs"]

    chunks = []
    timestamps = []

    for paragraph in paragraphs:
        chunk = ""
        timestamp = paragraph["start"]
        hour = int(timestamp//3600)
        mins = int((timestamp%3600)//60)
        seconds = int((timestamp%3600)%60)
        if hour: timestamp = f"{hour}:{mins:02d}:{seconds:02d}"
        else: timestamp = f"{mins:02d}:{seconds:02d}"
        for j, sentence in enumerate(paragraph["sentences"]):
            chunk+=sentence["text"]+" "
        chunks.append(chunk)
        timestamps.append(timestamp)

    dct = {"page":timestamps, "text":chunks}
    df = pd.DataFrame(dct)
    df.to_csv("texts.csv", index=False)