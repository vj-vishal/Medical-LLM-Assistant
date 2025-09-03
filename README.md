# 🧠Medical-LLM-Assistant
**Fine-tuned LLaMA Model with RAG for Medical Q&amp;A**

This repository contains code for fine-tuning LLaMA-3.2-1B-Instruct using Unsloth and integrating a Retrieval-Augmented Generation (RAG) pipeline with ChromaDB for evidence-based medical responses.
A Streamlit interface is also provided for interactive use.

<img width="767" height="868" alt="Image" src="https://github.com/user-attachments/assets/d6ec94b1-4bf8-4da5-b64b-b1ebc9965b5c" />

## 🚀 Features
- **LLM Fine-tuning** with LoRA adapters using [Unsloth](https://github.com/unslothai/unsloth)
- **Retrieval-Augmented Generation (RAG)** for contextual, evidence-supported answers
- **Medical PDF processing** and vector storage using **ChromaDB**
- **Interactive UI** built with **Streamlit** for easy question-answering

## 🏗️ Project Architecture

This project has **three main parts**:

---

### **1️⃣ Fine-Tuning**
- Fine-tunes the **LLaMA-3.2-1B model** using **LoRA** and **QLoRA** for efficiency.
- Saves checkpoints and the final trained model in **GGUF** format for inference.

**Flow:**  
Dataset → Tokenization → LoRA Fine-Tuning → Save Model → GGUF Export

---

### **2️⃣ RAG (Retrieval-Augmented Generation)**
- Processes medical PDFs.
- Splits documents into small chunks.
- Generates embeddings and stores them in **ChromaDB**.
- Retrieves the **most relevant evidence** for a given question.

**Flow:**  
PDF → Split Text → Create Embeddings → Store in ChromaDB → Retrieve Evidence

---

### **3️⃣ Inference & Streamlit UI**
- Takes **user questions** and **context**.
- Retrieves **supporting evidence** from the vector database.
- Generates a final **evidence-based answer** using the fine-tuned model.

**Flow:**  
Question + Context → Retrieve Evidence → Generate Answer → Display in Streamlit UI

---

### **Overall Workflow**
Dataset / PDF → Fine-Tuning → Vector Store → RAG → Streamlit Interface → Evidence-Based Answer

### **Steps to Use**

1. Upload your **medical PDF**  
2. Type your **question** and **context**  
3. Click **Get Answer** to see model-generated, evidence-based responses  

### 🛠 **Technologies Used**

- [Unsloth](https://github.com/unslothai/unsloth) for efficient LLaMA fine-tuning  
- [Transformers](https://huggingface.co/docs/transformers) for LLM support  
- [ChromaDB](https://www.trychroma.com/) for vector storage and retrieval  
- [LangChain](https://www.langchain.com/) for document splitting  
- [Streamlit](https://streamlit.io/) for interactive UI  

### 📈 **Future Improvements**

- Add multi-PDF ingestion support  
- Deploy as a web service with Docker and FastAPI  
- Expand medical dataset for better accuracy
- Fine-tune larger models (e.g., LLaMA-3-8B, LLaMA-3-70B, or Mistral-7B) for improved reasoning and medical context understanding   


### Set-up & Execution

1. Run the following command to install all dependencies. 

    ```bash
    pip install -r app/requirements.txt
    ```

1. Run the streamlit app by running the following command.

    ```bash
    streamlit run app/main.py
    ```

## 📜 License  

This project is licensed under the **Apache License 2.0**
