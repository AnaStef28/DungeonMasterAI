import os
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from llama_cpp import Llama


def split_into_chunks(text, max_words=100, overlap=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words - overlap):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks


def load_texts_from_folder(folder_path):
    chunks = []

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        if filename.endswith('.txt') or filename.endswith('.md'):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                chunks.extend(split_into_chunks(text))

        elif filename.endswith('.csv'):
            df = pd.read_csv(filepath)
            for _, row in df.iterrows():
                row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
                chunks.extend(split_into_chunks(row_text))

        elif filename.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

                if isinstance(data, list):
                    for item in data:
                        item_text = json.dumps(item, ensure_ascii=False)
                        chunks.extend(split_into_chunks(item_text))
                elif isinstance(data, dict):
                    item_text = json.dumps(data, ensure_ascii=False)
                    chunks.extend(split_into_chunks(item_text))

    return chunks


def build_prompt(context, question):
    context_text = "\n".join(context)
    return f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"


def retrieve_context(query, embedder, index, metadata, top_k=3):
    query_vec = embedder.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_vec), top_k)
    context=''
    for i in indices[0]:
        context+=metadata[i]
    return context


def prepare_embeddings():
    # === Load model and data ===
    print("Loading embedding model...")
    embedder = SentenceTransformer('paraphrase-MiniLM-L3-v2')

    folder_path = 'D:\\Facultate\\Python\\FML\\Project\\Funny_FML\\Data'
    print(f"Loading and embedding data from: {folder_path}")
    passages = load_texts_from_folder(folder_path)
    embeddings = embedder.encode(passages)

    # Build FAISS index
    print("Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    metadata = passages

    return embedder, index, metadata


# print("Loading LLaMA model...")
# llm = Llama(model_path="C:\\Users\\Ana\\.lmstudio\\models\\NousResearch\\Hermes-3-Llama-3.2-3B-GGUF\\Hermes-3-Llama-3.2-3B.Q4_K_M.gguf",
#             n_ctx=8192,
#             verbose=False)


