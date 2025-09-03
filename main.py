import streamlit as st
from rag import processPdf,merge_evidence,medical_inference

st.title("Medical Assistant")
st.markdown("Upload a medical document and ask questions to get evidence-based answers.")

placeholder= st.empty()

uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_pdf is not None:
    pdf_path = "temp.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.getbuffer())
        for status in processPdf(pdf_path):
            placeholder.write(status)
    Question = st.text_input("Type your question")
    Context = st.text_input("Type your context")
    if st.button("Get Answer", type="primary"):
        if Question and Context:
            try:
                evidence = merge_evidence(Question)
                answer = medical_inference(Question, Context, evidence)
                st.header("Answer:")
                st.write(answer)
                with st.expander("📚 Evidence Used"):
                            st.write(evidence)
            except RuntimeError as e:
                placeholder.write("You must ask Question and upload Context")
else:
    placeholder.write("You must upload a pdf")



