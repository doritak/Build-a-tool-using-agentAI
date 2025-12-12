import os
import json
import numpy as np
import streamlit as st

from dotenv import load_dotenv
from ebooklib import epub
from bs4 import BeautifulSoup
from openai import OpenAI
import faiss

# ==========================
# CONFIG & GLOBALS
# ==========================

st.set_page_config(page_title="Book RAG Agent", page_icon="📚")

# Load environment variables (.env for local, st.secrets for Streamlit Cloud)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY is missing. Set it in a .env file or in Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1-mini"

BOOK_PATH = "data/pg1342.epub"  # Local EPUB file
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


# ==========================
# EPUB PROCESSING & INDEXING
# ==========================

def extract_text_from_epub(path: str) -> str:
    """
    Extract readable text from a local .epub file.
    Some ebooklib versions don't expose epub.ITEM_DOCUMENT, so we use type 9 directly.
    """
    book = epub.read_epub(path)
    texts = []

    for item in book.get_items():
        # 9 corresponds to ITEM_DOCUMENT in ebooklib
        if item.get_type() == 9:
            soup = BeautifulSoup(item.get_body_content(), "lxml")
            # Remove scripts/styles
            for tag in soup(["script", "style"]):
                tag.extract()
            text = soup.get_text(separator=" ", strip=True)
            if text:
                texts.append(text)

    return "\n\n".join(texts)

def get_book_title(path: str) -> str:
    """
    Read the EPUB metadata and return the book title, if available.
    """
    book = epub.read_epub(path)
    titles = book.get_metadata("DC", "title")
    if titles:
        # titles is a list of (value, attributes) tuples
        return titles[0][0]
    return "Unknown title"

def chunk_text(text: str, chunk_size: int, overlap: int):
    """
    Split text into overlapping chunks so they fit better in context windows.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def embed_texts(texts):
    """
    Create embeddings for a list of strings using OpenAI.
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    vectors = [d.embedding for d in response.data]
    return np.array(vectors, dtype="float32")


class FaissIndex:
    """
    Simple FAISS wrapper storing chunks in memory.
    """

    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []

    def add(self, vectors: np.ndarray, chunks):
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int = 5):
        query_vector = np.expand_dims(query_vector, axis=0)
        distances, indices = self.index.search(query_vector, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                {
                    "text": self.chunks[idx],
                    "distance": float(dist),
                }
            )
        return results


def build_index() -> FaissIndex:
    """
    Load the local EPUB, chunk it, create embeddings and build a FAISS index.
    """
    if not os.path.exists(BOOK_PATH):
        raise FileNotFoundError(f"EPUB file not found at: {BOOK_PATH}")

    text = extract_text_from_epub(BOOK_PATH)
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        raise ValueError("No useful text could be extracted from the EPUB file.")

    embeddings = embed_texts(chunks)
    dim = embeddings.shape[1]

    index = FaissIndex(dim)
    index.add(embeddings, chunks)
    return index


# ==========================
# TOOLS (FUNCTION-CALLING)
# ==========================

def tool_search_in_book(query: str, index: FaissIndex, k: int) -> str:
    """
    Implementation of the 'search_in_book' tool.
    It searches in the FAISS index and returns the best matching chunks.
    Returns:
      - tool_text: formatted text to send to the LLM as tool content
      - results: raw list of chunks with distances
    """
    query_embedding = embed_texts([query])[0]
    results = index.search(query_embedding, k=k)

    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[Chunk {i} | distance={r['distance']:.4f}]\n{r['text']}")
    tool_text = "\n\n---\n\n".join(blocks)
    return tool_text, results


TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "search_in_book",
            "description": "Search relevant passages inside the book using a vector index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "User query or search phrase about the book.",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


def run_agent(user_message: str, index: FaissIndex) -> str:
    """
    Agent that uses:
    - OpenAI ChatCompletion as the LLM
    - A custom tool 'search_in_book' backed by FAISS
    - The current answer language selected in the UI
    """
    # Current language from UI
    lang_code = st.session_state.get("language", "en")

    lang_names = {
        "en": "English",
        "es": "Spanish",
        "de": "German",
        "ru": "Russian",
    }
    lang_text = lang_names.get(lang_code, "English")

    # system_message = (
    #     "You are an assistant specialized in answering questions about a specific book.\n"
    #     "You have access to a tool called 'search_in_book' that lets you fetch relevant "
    #     "passages from the book via a vector index.\n"
    #     f"The user can ask questions in any language, but you must ALWAYS answer in {lang_text}.\n"
    #     "If you need information from the book, call the 'search_in_book' tool first, "
    #     "then use its results to craft your final answer."
    # )
    system_message = (
        "You are an assistant specialized in answering questions about a specific book.\n"
        "You have access to a tool called 'search_in_book' that returns relevant passages.\n"
        "You MUST base your answers ONLY on the content returned by that tool.\n"
        "If the tool does not provide enough information to answer, you MUST say "
        "'I don't know based on the book' and explain briefly.\n"
        f"The user can ask in any language, but you must ALWAYS answer in {lang_text}."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    # First call: let the model decide whether to use a tool
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=TOOLS_SPEC,
        tool_choice="auto",
    )

    msg = response.choices[0].message

    # If the model did not call any tool, return its answer directly
    if not msg.tool_calls:
        return msg.content

    # If there are tool calls, handle them (we only have one tool here)
    messages.append(msg)  # append assistant message containing tool_calls

    for tool_call in msg.tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        if tool_name == "search_in_book":
            k_chunks = st.session_state.get("k_chunks", 5)
            tool_text, tool_result = tool_search_in_book(
                query=tool_args["query"],
                index=index,
                k=k_chunks,
            )
            # aqui guardamos los chunks para mostrarlos en la UI
            st.session_state.last_chunks = tool_result
            # Add the tool result back into the conversation
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": tool_text,
                }
            )

    # Second call: the model now has tool results and can answer in the selected language
    final_response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
    )

    return final_response.choices[0].message.content


# ==========================
# STREAMLIT UI
# ==========================

# Initialize session state keys
if "index" not in st.session_state:
    st.session_state.index = None

if "language" not in st.session_state:
    st.session_state.language = "en"  # default language: English

if "history" not in st.session_state:
    st.session_state.history = []

st.title("📚 Book RAG Agent with Tools")
st.write(
    f"This app loads a local EPUB book (`{BOOK_PATH}`), builds a vector index, "
    "and lets you ask questions to an LLM that uses a custom tool to search the book."
)

# ----- Language selector -----
LANG_OPTIONS = {
    "🇬🇧 English": "en",
    "🇪🇸 Spanish": "es",
    "🇩🇪 German": "de",
    "🇷🇺 Russian": "ru",
}

st.subheader("Answer language")

selected_label = st.radio(
    "Select the language in which the agent should answer:",
    list(LANG_OPTIONS.keys()),
    index=0,          # default: English
    horizontal=True,  # radio buttons in a row
)
st.subheader("Number of chunks to retrieve from vector DB")
# esto es un slider para seleccionar el numero de chunks a recuperar del vector DB
# k_value = st.slider(
#     "Select number of chunks:",
#     min_value=2,
#     max_value=5,
#     value=5,
#     step=1,
# )
# esto es un radio button para seleccionar el numero de chunks a recuperar del vector DB
k_value = st.radio(
    "Select how many chunks the agent should use:",
    options=[2, 3, 4, 5],
    index=3,  # default: 5
    horizontal=True  # show side-by-side
)
# store in session_state
st.session_state.k_chunks = k_value
st.session_state.language = LANG_OPTIONS[selected_label]

st.markdown(f"**Current answer language:** `{st.session_state.language}`")

# ----- Build index once -----
if st.session_state.index is None:
    with st.spinner("Loading the book and building the FAISS index..."):
        try:
            st.session_state.index = build_index()
            st.session_state.book_title = get_book_title(BOOK_PATH)
            st.success("✅ Book loaded and index built successfully.")
        except Exception as e:
            st.error(f"Error while building the index: {e}")
            st.stop()

# ----- Chat-like CLI -----
st.subheader(f"💬 Ask a question about the book: {st.session_state.book_title}")

user_query = st.text_input(
    "Type your question and press Enter:",
    placeholder="For example: Who is the main character?",
)

if user_query:
    with st.spinner("Thinking..."):
        answer = run_agent(user_query, st.session_state.index)
        st.session_state.history.append({"q": user_query, "a": answer})

# Show history (latest first)
# aqui me muestra el historial de preguntas y respuestas todas
# for turn in reversed(st.session_state.history):
#     st.markdown(f"```shell\n> {turn['q']}\n```")
#     st.markdown(f"**Answer:**\n\n{turn['a']}")
#     st.markdown("---")

# Show only the most recent answer
if st.session_state.history:
    last = st.session_state.history[-1]
    st.markdown(f"**Answer:**\n\n{last['a']}")
# Show the source chunks used for the last answer
if "last_chunks" in st.session_state and st.session_state.last_chunks:
    with st.expander("Show source chunks from vector database"):
        for i, r in enumerate(st.session_state.last_chunks, start=1):
            st.markdown(f"**Chunk {i} (distance = {r['distance']:.4f})**")
            st.write(r["text"])
            st.markdown("---")