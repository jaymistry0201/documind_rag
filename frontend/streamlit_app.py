import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
BACKEND_URL = os.getenv("DOCUMIND_BACKEND_URL", "http://127.0.0.1:8000")

# -------------------------------- Streamlit Page Setup ---------------------------------
# Set up page title, icon, and layout
st.set_page_config(
    page_title="DocuMind - PDF Q&A",
    page_icon="📄",
    layout="wide"
)

# Page header
st.title("📄 DocuMind — AI Powered PDF Query Assistant")
st.markdown("---")

# ------------------------------- Sidebar for File Upload -------------------------------
with st.sidebar:
    st.header("📤 Upload Your PDFs")
    st.markdown("Securely upload one or more documents (Limit 200MB each).")
    # File uploader allowing multiple PDF uploads
    uploaded_files = st.file_uploader(
        "PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    st.markdown("---")

    # Sidebar info for user guidance
    st.info(
        "💡 **How it works:**\n"
        "1. Upload your PDFs.\n"
        "2. DocuMind processes them using RAG technology.\n"
        "3. Ask questions and get answers grounded in your documents."
    )

# ----------------------------------- Main Logic -----------------------------------------
# If no files uploaded, show a friendly placeholder message
if not uploaded_files:
    st.markdown(
        """
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <h3 style="color: #4a5568;">Waiting for Documents...</h3>
            <p style="color: #718096;">Please upload PDFs using the control in the left sidebar to begin querying.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    # --------------------------------- State Management ----------------------------------
    # Maintain uploaded files across reruns
    if 'uploaded_file_ids' not in st.session_state:
        st.session_state['uploaded_file_ids'] = []
    if 'uploaded_file_names' not in st.session_state:
        st.session_state['uploaded_file_names'] = []


    # Only consider new files that haven't been uploaded yet
    new_files = [f for f in uploaded_files if f.name not in st.session_state['uploaded_file_names']]
    if new_files:
        # Prepare files for backend POST request
        files_to_upload = [("files", (f.name, f, "application/pdf")) for f in new_files]

        try:
            with st.spinner("Uploading and processing PDFs..."):
                # Send files to backend for processing
                resp = requests.post(f"{BACKEND_URL}/upload_pdfs", files=files_to_upload)
                resp.raise_for_status()
                data = resp.json()

            # Update session state with new file info
            for f in data["uploaded"]:
                st.session_state['uploaded_file_ids'].append(f["file_id"])
                st.session_state['uploaded_file_names'].append(f["filename"])
                st.success(f"Uploaded {f['filename']} with {f['num_chunks']} chunks.")

        except requests.exceptions.RequestException as e:
            st.error(f"🚨 Error uploading PDFs: {e}")

    # ----------------------------------------- Q&A Interface -------------------------------
    # Text input for user query
    query = st.text_input("💬 Ask a question about the uploaded PDFs:", key="query_input")

    # Button to generate answer
    if st.button("Generate Answer", use_container_width=True, type="primary"):
        if query.strip() == "":
            st.warning("Please enter a question to query your PDFs.")
        else:
            try:
                with st.spinner("🧠 Analyzing Documents and Generating Response..."):
                    # Send query to backend with optional top_k parameter
                    response = requests.post(
                        f"{BACKEND_URL}/query",
                        data={"query": query, "top_k": 5}
                    )
                    response.raise_for_status()
                    answer_data = response.json()

                # ------------------------------- Display Results --------------------------
                if "answer" in answer_data:
                    st.markdown("---")
                    st.markdown("### 💡 RAG-Powered Answer:")
                    st.markdown(answer_data["answer"])

                    # Nicely formatted citations for multiple PDFs
                    if "citations" in answer_data and len(answer_data["citations"]) > 0:
                        st.markdown("### Grounding Citations:")
                        citation_container = st.expander(f"View {len(answer_data['citations'])} Source Snippets")
                        with citation_container:
                            for c in answer_data["citations"]:
                                source = c.get("source", "Unknown File")
                                page = c.get("page", "?")
                                rank = c.get("rank", "?")
                                score = c.get("score", 0.0)

                                st.markdown(f"**{source}**")
                                st.markdown(f"Page {page} — (Rank: {rank}, Score: {score:.4f})\n")

                    else:
                        st.info("No citations found for this query.")

                elif "error" in answer_data:
                    st.error("❌ Backend Error: " + answer_data["error"])
                else:
                    st.warning("⚠️ Unexpected backend response format.")

            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to backend: {e}")
