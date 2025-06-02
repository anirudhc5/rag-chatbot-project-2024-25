import os
from flask import Flask, request, render_template, send_from_directory, jsonify, session, redirect
import openai
from openai import OpenAI
from werkzeug.utils import secure_filename
from parser import parse_pdf, parse_img, resetpages
from chatbot import create_embeddings, get_closest_embedding, get_rag_response
from voiceparser import parse_audio
from PIL import Image
import pymupdf
import json

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"pdf","jpg","png","jpeg","mp3"}
openai.api_key = os.environ.get("OPENAI_API_KEY", "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")

ai_client = OpenAI(api_key=openai.api_key)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

embeddings = []
texts = []
pages = []
audiobook_used = False
global filedir

def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def img_to_pdf(image_paths, out_pdf):
    imgs = [Image.open(p).convert("RGB") for p in image_paths]
    imgs[0].save(out_pdf, save_all=True, append_images=imgs[1:])

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    global embeddings, texts, pages, audiobook_used, filedir
    filedir = ""
    resetpages()
    session.pop("history", None)
    audiobook_used = False
    if os.path.isfile("texts.csv"): os.remove("texts.csv")
    if "fileUpload" not in request.files:
        return "No file part", 400
    files = request.files.getlist("fileUpload")
    pdf_paths, image_paths = [], []
    for file in files:
        if not file or not file.filename:
            return "No selected file", 400
        extension = file.filename.rsplit(".",1)[1].lower()
        filename  = secure_filename(file.filename)
        dst = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(dst)
        if extension == "pdf":
            parse_pdf(dst)
            pdf_paths.append(filename)
        elif extension in {"png","jpg","jpeg"}:
            parse_img(dst)
            image_paths.append(filename)
        elif extension == "mp3":
            filedir = filename
            audiobook_used = True
            parse_audio(dst)
        else:
            return "Invalid file type", 400
    embeddings, texts, pages = create_embeddings("texts.csv")
    if not filedir:
        if pdf_paths or image_paths:
            if image_paths:
                img_pdf = os.path.join(app.config["UPLOAD_FOLDER"], "images.pdf")
                full_imgs = [os.path.join(app.config["UPLOAD_FOLDER"], i) for i in image_paths]
                img_to_pdf(full_imgs, img_pdf)
                pdf_paths.append("images.pdf")
            doc = pymupdf.open()
            for filename in pdf_paths:
                part = pymupdf.open(os.path.join(app.config["UPLOAD_FOLDER"], filename))
                doc.insert_pdf(part)
                part.close()
            filedir = "textbooks.pdf"
            doc.save(os.path.join(app.config["UPLOAD_FOLDER"], filedir))
            doc.close()
    return redirect("/chat")

@app.route("/chat")
def chat():
    global filedir
    return render_template("chat.html", filename=filedir)

@app.route("/uploaded/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/get_history")
def get_history():
    return jsonify({"history": session.get("history", [])})

@app.route("/clear_history", methods=["POST"])
def clear_history():
    session.pop("history", None)
    return jsonify({"status":"cleared"})

@app.route("/ask_chatbook", methods=["POST"])
def ask_chatbook():
    question = request.form.get("question")
    history  = session.get("history", [])
    rephrased = ai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
            "role":"system",
            "content": "This is based on a user-uploaded textbook. Interpret requests like 'formula' based on the content type. If it's a math or physics concept, return the scientific or mathematical formula, not a chemical one. Same applies with other words."
            },
            {
            "role":"user",
            "content": f"Rephrase the following question to be more detailed and clearer:\n\n{question}"
            }
        ]
    ).choices[0].message.content
    sims_texts, sims_pages = get_closest_embedding(rephrased, embeddings, texts, pages)
    answer = get_rag_response(rephrased,sims_texts, sims_pages, original_question=question, filetype="mp3" if audiobook_used else "pdf", history=history)
    history.append({"role":"user", "content": question})
    history.append({"role":"assistant", "content": answer})
    session["history"] = history
    return jsonify({"history": history})

@app.route("/generate_flashcards/<int:num>/<topic>")
def generate_flashcards(num, topic):
    closest_texts, closest_pages = get_closest_embedding(topic, embeddings, texts, pages, top_k=5)
    context = "\n".join(f"[Page {p}] {t}" for p,t in zip(closest_pages, closest_texts))
    prompt = f"""
    You are Chatbook’s flashcard generator.
    Using *only* the following context, generate exactly {num} flashcards on the topic "{topic}".  
    {context}
    Keep responses short.
    **Output** **only** a JSON array. Each element must be an object with exactly two keys like so (the example below is a length 1 JSON object):
    [
    {{"question": "question goes here", "answer": "answer goes here"}}
    ]
    **Do not** output any other text or formatting. Especially avoid "json" in the beginning and backslash strings e.g. "\\x02".
    """
    response = ai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"system","content":prompt}]
    )
    deck = json.loads(response.choices[0].message.content)
    return jsonify({"flashcards": deck})

if __name__=="__main__":
    app.run(host="0.0.0.0", port="8080", debug=True)
