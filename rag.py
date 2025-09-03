import chromadb
from pathlib import Path
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
import ollama

path = Path(__file__).parent/"Resource/vectorstore"
chroma_client= chromadb.PersistentClient(path=str(path))
collection_name_rag='rags'
CHUNK_SIZE=500
CHUNK_OVERLAP= 25

ef=embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="aleynahukmet/bge-medical-small-en-v1.5"
)

vector_store= None

def processPdf(pdf_path):
    global vector_store

    yield "reseting vector store"
    try:
        chroma_client.delete_collection(collection_name_rag)
    except ValueError:
        pass

    vector_store = chroma_client.create_collection(
        name=collection_name_rag,
        embedding_function=ef
    )

    yield "loading pdf"
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    yield "Splitting text into chunks...✅"
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
                                              separators=["\n\n", "\n", ".", " "])

    chunks = splitter.split_documents(documents)

    docs = [chunk.page_content for chunk in chunks]
    metadata = [chunk.metadata for chunk in chunks]
    uuids = [str(uuid4()) for _ in range(len(chunks))]

    yield "Adding document to vector_store...✅"
    vector_store.add(
        documents=docs,
        metadatas=metadata,
        ids=uuids
    )
    yield "Documents added. You can ask question now"

def get_relevant_evidence(question):
    if vector_store is None:
        raise ValueError("Vector store not initialized. Call processPdf() first.")
    result = vector_store.query(
        query_texts=[question],
        n_results=3
    )
    return result

def merge_evidence(question):
    result= get_relevant_evidence(question)
    evidence_topk = [r for r in result["documents"][0]]
    evidence = "\n\n".join(evidence_topk)
    return evidence

def medical_inference(question, context, evidence):
    sys_prompt = """You are a reflective assistant engaging in thorough, iterative reasoning, mimicking human stream-of-consciousness 
    thinking. Your approach emphasizes exploration, self-doubt, and continuous refinement before coming up with an answer.

    RULES:
    - Address the exact question asked
    - Refer specific information from the context
    - Acknowledge when evidence is limited
    - Avoid contradictory statements
    - Use context as patient history and evidence as supporting documents while answering the question.
    - If evidence contradicts or is insufficient, state this clearly
    
    <input>
    QUESTION:
    {}
    
    CONTEXT:
    {}
    
    EVIDENCE:
    {}
    
    </input>
    ANSWER:
    """

    full_prompt = sys_prompt.format(question, context, evidence)

    response = ollama.generate(
        model='finetuned-model',
        prompt=full_prompt,
        options={
            'temperature': 0.1,
            'num_ctx': 2096,
            'num_predict': 1024
        }
    )

    return response['response']

if __name__== "__main__":
    pdf_path = Path(__file__).parent / "vitaminb12-consumer.pdf"
    processPdf(pdf_path)
    question="Do vitamin B12 shots cause derealization symptoms?"
    context = """Hi doctor, I have been suffering from the derealization symptoms for the past 5 years. I believe they were caused by B12 shots I was
    taking to treat autism. After my brain scan I got to know that my inhibition was too high. I do not find any safety net to keep my mind from 
    deconstructing things. A part of me have a dislike for anything involving neuroplasticity. There are too much dichromatic thinking and lack of 
    magical thinking. Currently, I am on Adderall and Prozac."""
    evidence= merge_evidence(question)
    answer= medical_inference(question, context, evidence)
    print(answer)
