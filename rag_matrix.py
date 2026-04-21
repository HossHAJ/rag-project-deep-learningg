# ============================================================
# RAG Project - Centrale Lyon Deep Learning
# ============================================================



from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

# ============================================================
# 1. Load and split PDF
# ============================================================

loader = PyPDFLoader("documatrix.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)
text_lines = [chunk.page_content for chunk in chunks]

print(f"Total chunks: {len(text_lines)}")

# ============================================================
# 2. Embedding model
# ============================================================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def emb_text(text):
    return embedding_model.encode(
        [text],
        normalize_embeddings=True
    ).tolist()[0]

# ============================================================
# 3. ChromaDB
# ============================================================

client = PersistentClient(path="./my_db.db")
collection_name = "rag_collection"

try:
    collection = client.get_collection(name=collection_name)
    print("Loaded existing collection.")
except:
    collection = client.create_collection(name=collection_name)
    print("Created new collection. Embedding documents...")

    embeddings = []
    ids = []

    for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
        embeddings.append(emb_text(line))
        ids.append(str(i))

    collection.add(
        documents=text_lines,
        embeddings=embeddings,
        ids=ids
    )

    print("Done adding documents.")

# ============================================================
# 4. Prompt template
# ============================================================

PROMPT = """
Answer the question using only the context below.

Context:
{context}

Question:
{question}

Answer in one short paragraph:
"""

# ============================================================
# 5. Load FLAN-T5 directly
# ============================================================

print("\nLoading LLM...")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def ask_rag(question, n_results=3):
    query_embedding = emb_text(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    context = "\n".join(results["documents"][0])

    prompt = PROMPT.format(
        context=context,
        question=question
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer, context

# ============================================================
# 6. First test
# ============================================================

question = "Who is Neo and what is the Matrix?"
answer, context = ask_rag(question)

print("\n--- Retrieved Context ---")
print(context[:500], "...")

print("\n--- Question ---")
print(question)

print("\n--- Answer ---")
print(answer)

# ============================================================
# 7. Interactive loop
# ============================================================

#print("\n=== RAG System Ready ===")
#print("Type your question about the document (or 'quit' to exit)\n")

#while True:
    #user_question = input("Your question: ").strip()

    #if user_question.lower() in ("quit", "exit", "q"):
        #break

    #if not user_question:
       # continue

    #response, _ = ask_rag(user_question)
    #print(f"\nAnswer: {response}\n")

# ============================================================
# --- 8. Gradio Interface ---
# ============================================================

import gradio as gr

def ask_interface(question):
  
  
    answer, _ = ask_rag(question)
    return answer


demo = gr.Interface(
    fn=ask_interface,
    inputs=gr.Textbox(
        lines=2,
        placeholder="Ask a question about the document..."
    ),
    outputs="text",
    title="RAG Question Answering System",
    description="Ask questions about the document using a Retrieval-Augmented Generation system."
)

if __name__ == "__main__":
    print("\nLaunching Gradio interface...")
    demo.launch()