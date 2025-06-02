import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from sklearn.metrics.pairwise import cosine_similarity

openai_api_key = os.environ.get("OPENAI_API_KEY", "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")
client = OpenAI(api_key=openai_api_key)

def get_embedding(text):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode([text], batch_size=1, show_progress_bar=False)[0]

def create_embeddings(filepath):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    df = pd.read_csv(filepath)
    texts = df["text"].tolist()
    pages = df["page"].tolist()
    embs  = [model.encode(t, show_progress_bar=False) for t in texts]
    return embs, texts, pages

def get_closest_embedding(question, embeddings, texts, pages, top_k=5):
    q_vec = get_embedding(question)
    sims  = cosine_similarity([q_vec], embeddings)[0]
    idxs  = np.argsort(sims)[-top_k:][::-1]
    return [texts[i] for i in idxs], [pages[i] for i in idxs]

def get_rag_response(user_question, snippets, snippet_pages, original_question, filetype="pdf", history=None):
    context = "\n".join(f"[{p}] {s}" for p,s in zip(snippet_pages, snippets))

    system_prompt = ""
    if filetype == "pdf":
        system_prompt = (
            "You are Chatbook, a tutoring assistant using the Feynman Technique to explain concepts clearly and in detail. "
            "You may only use the provided excerpts to answer. "
            "Your goal is to explain the content to a high school student with no prior background. "
            "Use step-by-step reasoning, and use simple language. "
            "For every fact you state, cite the page number the text came from by appending it in brackets, e.g. [12], "
            "at the end of the sentence. "
            "Do not add external knowledge or hallucinate."
            "Wrap all LaTeX math expressions in your responses using dollar signs: use $$...$$ for display equations and $...$ for inline math."
        )
    else:
        system_prompt = (
            "You are Chatbook, a tutoring assistant using the Feynman Technique to explain concepts clearly and in detail. "
            "You may only use the provided excerpts to answer. "
            "Your goal is to explain the content to a high school student with no prior background. "
            "Use step-by-step reasoning, and use simple language. "
            "For every fact you state, cite the timestamps the text came from by appending the approximate range of timestamps in brackets, e.g. [12:40 - 13:10], "
            "at the end of the sentence. "
            "Do not add external knowledge or hallucinate."
            "Wrap all LaTeX math expressions in your responses using dollar signs: use $$...$$ for display equations and $...$ for inline math."
        )

    messages = [{"role":"system","content": system_prompt}]

    if history:
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    messages.append({
        "role":"system",
        "content": "Document excerpts:\n" + context
    })

    messages.append({"role":"user","content": f"""
    Original question: {original_question}\n
    Rephrased question: {user_question}\n
    Use your answer for the rephrased question to answer the original question.
    """})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    return response.choices[0].message.content.strip()
