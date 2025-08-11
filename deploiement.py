from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.datasets import imdb

import tkinter as tk
from tkinter import scrolledtext, messagebox
import re

MODEL_PATH = "modele_imdb_bilstm.h5"   
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Impossible de charger le modèle '{MODEL_PATH}': {e}")


num_words = 10000  
maxlen = 200

word_index = imdb.get_word_index()
word_to_id = {k: (v + 3) for k, v in word_index.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
word_to_id["<UNUSED>"] = 3

max_words = num_words

def tokenize(text):
    """Tokenisation simple : lowercase + suppression ponctuation + split."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9àâäéèêëïîôöùûüç'-]", " ", text)
    tokens = [t for t in text.split() if t]
    return tokens

def encode_text(text):
    """Convertit une phrase en liste d'indices compatible IMDB."""
    tokens = tokenize(text)
    encoded = []
    for word in tokens:
        idx = word_to_id.get(word, word_to_id["<UNK>"])
        if idx >= max_words:
            idx = word_to_id["<UNK>"]
        encoded.append(idx)
    return encoded

def predict_sentiment(text):
    """Retourne (label_int, label_text, score_float)."""
    encoded = encode_text(text)
    padded = pad_sequences([encoded], maxlen=maxlen)
    pred = model.predict(padded, verbose=0)[0]
    try:
        score = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
    except:
        score = float(pred)
    label_int = 1 if score > 0.5 else 0
    label_text = "Positif" if label_int == 1 else "Négatif"
    return label_int, label_text, score

class SentimentChatApp:
    def __init__(self, root):
        self.root = root
        root.title("Chat Sentiment (IMDB)")

        self.chat_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', width=70, height=20)
        self.chat_box.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        self.entry = tk.Entry(root, width=55)
        self.entry.grid(row=1, column=0, padx=(10,0), pady=(0,10))
        self.entry.bind("<Return>", lambda event: self.on_send())

        send_btn = tk.Button(root, text="Envoyer", command=self.on_send, width=12)
        send_btn.grid(row=1, column=1, padx=5, pady=(0,10))

        clear_btn = tk.Button(root, text="Effacer", command=self.clear_chat, width=12)
        clear_btn.grid(row=1, column=2, padx=(0,10), pady=(0,10))

        self.chat_box.tag_config("user", foreground="#0b486b", justify='left')
        self.chat_box.tag_config("bot_pos", foreground="#1b7a1a", justify='left')   
        self.chat_box.tag_config("bot_neg", foreground="#b22222", justify='left')   
        self.chat_box.tag_config("meta", foreground="#555555", font=("Helvetica", 8))

        self._append_bot("Bonjour ! Écris une critique de film et j'évaluerai le sentiment (0 = Négatif, 1 = Positif).", meta=True)

    def _append_user(self, text):
        self.chat_box.configure(state='normal')
        self.chat_box.insert(tk.END, f"Vous: {text}\n", "user")
        self.chat_box.configure(state='disabled')
        self.chat_box.see(tk.END)

    def _append_bot(self, text, label=None, score=None, meta=False):
        self.chat_box.configure(state='normal')
        prefix = "Bot: "
        if meta:
            self.chat_box.insert(tk.END, f"{text}\n", "meta")
        else:
            if label is not None:
                tag = "bot_pos" if label == 1 else "bot_neg"
                self.chat_box.insert(tk.END, f"{prefix}{text}\n", tag)
                if score is not None:
                    self.chat_box.insert(tk.END, f"    → label: {label}    score: {score:.3f}\n", "meta")
            else:
                self.chat_box.insert(tk.END, f"{prefix}{text}\n")
        self.chat_box.configure(state='disabled')
        self.chat_box.see(tk.END)

    def on_send(self):
        user_text = self.entry.get().strip()
        if not user_text:
            return
        self.entry.delete(0, tk.END)
        self._append_user(user_text)

        try:
            label_int, label_text, score = predict_sentiment(user_text)
            bot_text = f"{label_text}"
            self._append_bot(bot_text, label=label_int, score=score)
        except Exception as e:
            self._append_bot(f"Erreur lors de la prédiction : {e}", meta=True)
            messagebox.showerror("Erreur", f"Impossible de prédire : {e}")

    def clear_chat(self):
        self.chat_box.configure(state='normal')
        self.chat_box.delete('1.0', tk.END)
        self.chat_box.configure(state='disabled')
        self._append_bot("Conversation effacée. Écris une critique de film et j'évaluerai le sentiment (0 = Négatif, 1 = Positif).", meta=True)

if __name__ == "__main__":
    root = tk.Tk()
    root.resizable(False, False)
    app = SentimentChatApp(root)
    root.mainloop()
