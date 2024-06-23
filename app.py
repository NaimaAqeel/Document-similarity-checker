import os
import fitz  # PyMuPDF for PDF extraction
import docx  # python-docx for DOCX extraction
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, UploadFile
import gradio as gr

app = FastAPI()

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return ""

def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {str(e)}")
        return ""

def calculate_cosine_similarity(doc1: str, doc2: str) -> float:
    # Combine paragraphs into larger chunks (every 5 sentences for example)
    def chunk_text(text, chunk_size=5):
        sentences = text.split('.')
        chunks = ['.'.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
        return chunks

    chunks1 = chunk_text(doc1)
    chunks2 = chunk_text(doc2)

    # Get embeddings for chunks
    embeddings1 = model.encode(chunks1, convert_to_tensor=True)
    embeddings2 = model.encode(chunks2, convert_to_tensor=True)

    # Calculate cosine similarities between all chunk pairs
    cosine_similarities = util.pytorch_cos_sim(embeddings1, embeddings2)

    # Calculate the mean of the max similarities for each chunk
    max_similarities1 = cosine_similarities.max(dim=1)[0]
    max_similarities2 = cosine_similarities.max(dim=0)[0]
    mean_similarity = (max_similarities1.mean() + max_similarities2.mean()) / 2.0

    return mean_similarity.item()

def similarity(file1, file2):
    text1 = extract_text_from_pdf(file1.name) if file1.name.endswith('.pdf') else extract_text_from_docx(file1.name)
    text2 = extract_text_from_pdf(file2.name) if file2.name.endswith('.pdf') else extract_text_from_docx(file2.name)
    return calculate_cosine_similarity(text1, text2)

# Create a Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Document Similarity Checker")
    file1 = gr.File(label="Upload Document 1")
    file2 = gr.File(label="Upload Document 2")
    output = gr.Textbox(label="Similarity Score")
    submit = gr.Button("Submit")
    
    submit.click(fn=similarity, inputs=[file1, file2], outputs=output)

# Use the GRADIO_SERVER_PORT environment variable, default to 7860 if not set
port = int(os.getenv('GRADIO_SERVER_PORT', 7860))
demo.launch(server_name="0.0.0.0", server_port=port)