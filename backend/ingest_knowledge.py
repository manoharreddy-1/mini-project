import os
from rag_engine import extract_and_chunk_pdf, build_vector_store

DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')

def ingest_data_folder():
    """
    Scans the backend/data folder for PDFs and ingests them into ChromaDB.
    """
    if not os.path.exists(DATA_FOLDER):
        print(f"Directory {DATA_FOLDER} does not exist. Creating it now...")
        os.makedirs(DATA_FOLDER, exist_ok=True)
        print("Please place your PDF syllabus/notes into the 'data' folder and run this script again.")
        return

    pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.pdf')]
    if not pdf_files:
        print("No PDF files found in the 'data' folder. Please add some and run again.")
        return

    print(f"Found {len(pdf_files)} PDFs in the 'data' directory. Beginning extraction...")
    
    all_chunks = []
    
    for filename in pdf_files:
        filepath = os.path.join(DATA_FOLDER, filename)
        
        # Simple metadata parsing from filename if formatted like: "Subject_Unit_1.pdf"
        # Otherwise, falls back to defaults.
        parts = filename.replace('.pdf', '').split('_')
        
        subject_name = parts[0] if len(parts) > 0 else "General"
        unit_number = parts[-1] if len(parts) > 1 and parts[-1].isdigit() else "1"
        
        print(f" -> Processing '{filename}' | Detected Subject: {subject_name} | Unit: {unit_number}")
        try:
            chunks = extract_and_chunk_pdf(filepath, subject_name, unit_number)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if all_chunks:
        print(f"\\nVectorizing {len(all_chunks)} total chunks into ChromaDB...")
        build_vector_store(all_chunks)
        print("Completed. The Knowledge Base is ready for the RAG platform.")
    else:
        print("Failed to extract any text chunks from the files.")

if __name__ == "__main__":
    ingest_data_folder()
