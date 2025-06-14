import sys
if sys.platform == 'linux': # Apply only on Linux where Streamlit Cloud runs
    try:
        __import__('pysqlite3')
        import sqlite3
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
        # st.toast("Successfully patched sqlite3 with pysqlite3 for ChromaDB!", icon="✅") # Optional: for debugging
    except ImportError:
        # st.toast("pysqlite3 not found, ChromaDB might use system SQLite.", icon="⚠️") # Optional: for debugging
        pass
import streamlit as st
# LangChain imports for the Study Buddy section
from langchain_google_genai import GoogleGenerativeAI as LangChainGoogleGenerativeAI # Alias for clarity
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader # Still useful for text-based PDFs
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain.chains import RetrievalQA # Not directly used for chat in this version
from langchain.prompts import PromptTemplate
# Standard Python imports
import os
import tempfile
import hashlib
import time

# --- OCR Specific Imports (using Gemini directly) ---
import google.generativeai as genai # This is the primary SDK for Gemini

# --- App Configuration & Title ---
st.set_page_config(page_title="YashrajAI", layout="wide")
st.title("ULTIMATE AI Study Helper")

# --- API Key Configuration ---
try:
    GEMINI_API_KEY = st.secrets.get("GOOGLE_API_KEY_GEMINI", os.getenv("GOOGLE_API_KEY_GEMINI"))
except (FileNotFoundError, KeyError):
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY_GEMINI")

if not GEMINI_API_KEY:
    st.error("🔴 Gemini API Key (GOOGLE_API_KEY_GEMINI) not found. Please set it. All features will be disabled.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# --- Initialize LLM and Embeddings ---
llm_studybuddy = None # For summaries, flashcards, practice questions
llm_qna = None      # For chat Q&A
embeddings_studybuddy = None
try:
    llm_studybuddy = LangChainGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0.5, google_api_key=GEMINI_API_KEY)
    llm_qna = LangChainGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17", temperature=0.7, google_api_key=GEMINI_API_KEY)
    embeddings_studybuddy = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="retrieval_document", google_api_key=GEMINI_API_KEY)
except Exception as e:
    st.sidebar.error(f"Error initializing Gemini models: {e}")

# --- Session State Management ---
if 'ocr_text_output' not in st.session_state:
    st.session_state.ocr_text_output = None
if 'ocr_file_name' not in st.session_state:
    st.session_state.ocr_file_name = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'processed_file_hash' not in st.session_state:
    st.session_state.processed_file_hash = None
if 'documents_for_direct_use' not in st.session_state:
    st.session_state.documents_for_direct_use = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_doc_chat_hash' not in st.session_state:
    st.session_state.current_doc_chat_hash = None
if 'last_used_sources' not in st.session_state: 
    st.session_state.last_used_sources = []


# =============================================
# SECTION 1: OCR PDF (Using Gemini Multimodal)
# =============================================
st.sidebar.markdown("---")
st.sidebar.header("📄 OCR Scanned PDF ")
ocr_uploaded_file = st.sidebar.file_uploader("Upload a scanned PDF for OCR", type="pdf", key="gemini_ocr_uploader")

def perform_ocr_with_gemini(pdf_file_uploader_object):
    try:
        st.sidebar.write("Uploading PDF...")
        uploaded_gemini_file = genai.upload_file(
            path=pdf_file_uploader_object,
            display_name=pdf_file_uploader_object.name,
            mime_type=pdf_file_uploader_object.type
        )
#        st.sidebar.write(f"File '{uploaded_gemini_file.display_name}' uploaded. URI: {uploaded_gemini_file.uri}. Mime Type: {pdf_file_uploader_object.type}")
        st.sidebar.write("Extracting text...")
        model_ocr = genai.GenerativeModel(model_name="gemini-2.5-flash-preview-04-17")
        prompt = [
            "Please perform OCR on the provided PDF document and extract all text content.",
            "Present the extracted text clearly. If there are multiple pages, try to indicate page breaks with something like '--- Page X ---' if possible, or just provide the continuous text.",
            "Focus solely on extracting the text as accurately as possible from the document.",
            uploaded_gemini_file 
        ]
        response = model_ocr.generate_content(prompt, request_options={"timeout": 600})
        try:
            genai.delete_file(uploaded_gemini_file.name)
            st.sidebar.write(f"Temporary file '{uploaded_gemini_file.display_name}' deleted from File API.")
        except Exception as e_delete:
            st.sidebar.warning(f"Could not delete temporary file from File API: {e_delete}")
        return response.text
    except Exception as e:
        st.sidebar.error(f"OCR Error: {e}")
        if 'uploaded_gemini_file' in locals() and hasattr(uploaded_gemini_file, 'name'):
            try: genai.delete_file(uploaded_gemini_file.name)
            except: pass
        return None

if ocr_uploaded_file is not None:
    if st.sidebar.button("✨ Perform OCR", key="gemini_ocr_button"):
        st.session_state.ocr_text_output = None 
        st.session_state.ocr_file_name = None
        with st.spinner("Performing OCR... This may take a while for large files."):
            extracted_text = perform_ocr_with_gemini(ocr_uploaded_file)
            if extracted_text:
                st.session_state.ocr_text_output = extracted_text
                st.session_state.ocr_file_name = f"ocr_output_{os.path.splitext(ocr_uploaded_file.name)[0]}.txt"
                st.sidebar.success("OCR Complete!")
            else:
                st.sidebar.error("OCR failed or no text was extracted.")

if st.session_state.ocr_text_output:
    st.sidebar.subheader("OCR Result:")
    st.sidebar.download_button(
        label="📥 Download OCR'd Text",
        data=st.session_state.ocr_text_output.encode('utf-8'),
        file_name=st.session_state.ocr_file_name,
        mime="text/plain",
        key="download_gemini_ocr"
    )
    with st.sidebar.expander("Preview OCR Text (First 1000 Chars)"):
        st.text(st.session_state.ocr_text_output[:1000] + "...")


# =============================================
# SECTION 2: Study Buddy Q&A and Tools
# =============================================
st.sidebar.markdown("---")
st.sidebar.header("🧠 Study Buddy Tools")
study_uploaded_file = st.sidebar.file_uploader(
    "Upload TEXT-READABLE PDF or TXT for Q&A, Summary, etc.", 
    type=["pdf", "txt"], 
    key="study_uploader",
    help="If your PDF is scanned, please use the 'OCR Scanned PDF' section above first and then upload the downloaded .txt file here."
)

if study_uploaded_file is not None and GEMINI_API_KEY and llm_studybuddy and embeddings_studybuddy:
    file_bytes = study_uploaded_file.getvalue()
    current_file_hash = hashlib.md5(file_bytes).hexdigest()

    if current_file_hash != st.session_state.processed_file_hash:
        st.sidebar.info(f"New file '{study_uploaded_file.name}' for AI. Processing...")
        st.session_state.vector_store = None
        st.session_state.documents_for_direct_use = None
        st.session_state.chat_history = [] 
        st.session_state.current_doc_chat_hash = current_file_hash 
        st.session_state.last_used_sources = []
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{study_uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file_path = tmp_file.name
            if study_uploaded_file.type == "application/pdf":
                loader = PyPDFLoader(tmp_file_path)
            else:
                loader = TextLoader(tmp_file_path, encoding='utf-8')
            documents = loader.load()
            if study_uploaded_file.type == "application/pdf" and (not documents or not any(doc.page_content.strip() for doc in documents)):
                st.sidebar.error("Uploaded PDF for AI has no extractable text. Use OCR section first for scanned PDFs.")
                os.remove(tmp_file_path)
                st.session_state.processed_file_hash = None
            else:
                st.session_state.documents_for_direct_use = documents
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
                texts = text_splitter.split_documents(documents)
                valid_texts = [text for text in texts if text.page_content and text.page_content.strip()]
                if not valid_texts:
                    st.sidebar.error("No valid text chunks after splitting for AI.")
                else:
                    with st.spinner("Creating embeddings for AI..."):
                        st.session_state.vector_store = Chroma.from_documents(documents=valid_texts, embedding=embeddings_studybuddy)
                    st.session_state.processed_file_hash = current_file_hash
                    st.sidebar.success(f"✅ '{study_uploaded_file.name}' ready for AI!")
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
        except Exception as e:
            st.sidebar.error(f"Error processing file: {e}")
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path): os.remove(tmp_file_path)
            st.session_state.vector_store = None
            st.session_state.documents_for_direct_use = None
            st.session_state.processed_file_hash = None
            st.session_state.chat_history = []
            st.session_state.current_doc_chat_hash = None
            st.session_state.last_used_sources = []

# --- Backend Function for Practice Question Generation ---
def generate_practice_questions_with_guidance(subject_name, document_text, example_qa_style_guide, llm):
    """Generates practice questions using the user-preferred prompt structure."""
    
    PRACTICE_QUESTION_PROMPT_TEMPLATE = """You are an expert AI assistant tasked with generating practice questions for a {subject_name} exam, based ONLY on the provided "Document Text". Your goal is to emulate the style, type, and difficulty of the "Example Questions and Answers" provided for style guidance.

Instructions:

1.  Carefully review the "Document Text".
2.  Carefully review the "Example Questions and Answers" to understand the desired style, question types, and answer format for {subject_name}.
3.  Generate as many new and distinct practice questions based on the "Document Text" as you can.
4.  The generated questions should be similar in nature to the provided examples.
5.  For each question you generate, provide an answer based *strictly* on the information within the "Document Text".
6.  Output Format:
    *   Each question-answer pair must be on a new line and also leave a line after each question.
    *   Separate the question from its answer using ">>" (two greater-than signs with no spaces around them).
    *   The entire output should be formatted in Markdown.
    *   Do NOT number the questions.

Example Questions and Answers for {subject_name} (Follow this style):
{example_questions_and_answers}
Document Text:
{document_text}

Generated Practice Questions for {subject_name} (question>>answer format):
"""
    
    formatted_prompt = PRACTICE_QUESTION_PROMPT_TEMPLATE.format(
        subject_name=subject_name,
        document_text=document_text,
        example_questions_and_answers=example_qa_style_guide if example_qa_style_guide.strip() else "No specific style examples provided by user. Generate general questions suitable for the subject, inferring common question types for the specified subject based on the document text."
    )
    
    try:
        # Configure safety settings to be less restrictive if needed, but be cautious
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        # For LangChain's GoogleGenerativeAI, safety_settings are passed during LLM initialization
        # If using direct genai.GenerativeModel().generate_content(), pass it there.
        # Here, llm is already initialized, so we rely on its default or initialized safety settings.
        # If you face blocking issues, you might need to adjust safety_settings at LLM init
        # or use a direct genai call if LangChain wrapper doesn't expose it easily for invoke.

        response = llm.invoke(formatted_prompt) # llm here is llm_studybuddy
        return response
    except Exception as e:
        # Check for specific Google API errors related to safety or blocking
        if "response was blocked" in str(e).lower() or "safety settings" in str(e).lower():
            st.warning("The response was blocked due to safety settings. Try rephrasing style guidance or check document content.")
            return "Response blocked due to safety settings. Please check your input or document content."
        return f"Error generating practice questions: {e}"


# --- Main Interaction Area for Study Buddy Tools ---
if st.session_state.get('vector_store') and st.session_state.get('documents_for_direct_use') and GEMINI_API_KEY and llm_qna and llm_studybuddy:
    st.markdown("---")
    if st.session_state.current_doc_chat_hash != st.session_state.processed_file_hash:
        st.session_state.chat_history = []
        st.session_state.current_doc_chat_hash = st.session_state.processed_file_hash
        st.session_state.last_used_sources = []
        
    header_file_name = "your document"
    if study_uploaded_file and hasattr(study_uploaded_file, 'name'):
        if st.session_state.processed_file_hash == hashlib.md5(study_uploaded_file.getvalue()).hexdigest():
            header_file_name = study_uploaded_file.name
            
    st.header(f"🛠️ Study Tools for: {header_file_name}")
    
    query_type_key_suffix = st.session_state.processed_file_hash or "default_study_tools"
    
    tool_options = ["Chat & Ask Questions", 
                    "Generate Practice Questions",
                    "Generate Flashcards (Term>>Definition)", 
                    "Summarize Document"]
    query_type = st.radio(
        "What do you want to do with the text-readable document?",
        tool_options,
        key=f"query_type_{query_type_key_suffix}"
    )

    if query_type == "Chat & Ask Questions":
        st.subheader("💬 Chat with your Document")
        for item in st.session_state.chat_history:
            role = item.get("role")
            content = item.get("content")
            sources = item.get("sources") 
            with st.chat_message(role):
                st.markdown(content)
                if role == "ai" and sources: 
                    with st.expander("📚 View Sources Used", expanded=False):
                        for i, source_doc in enumerate(sources):
                            page_label = source_doc.metadata.get('page', 'N/A')
                            st.caption(f"Source {i+1} (Page: {page_label}):")
                            st.markdown(f"> {source_doc.page_content[:300]}...") 
                            st.markdown("---")
        
        user_question = st.chat_input("Ask a follow-up question or a new question...", key=f"chat_input_{query_type_key_suffix}")

        if st.button("Clear Chat History", key=f"clear_chat_{query_type_key_suffix}"):
            st.session_state.chat_history = []
            st.session_state.last_used_sources = []
            st.rerun()

        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question, "sources": None})
            with st.chat_message("user"): 
                st.markdown(user_question)

            with st.spinner("Thinking..."):
                history_for_prompt_list = [f"Previous {item['role']}: {item['content']}" for item in st.session_state.chat_history[:-1]]
                history_for_prompt = "\n".join(history_for_prompt_list)
                
                retriever = st.session_state.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                
                prompt_template_chat_qa = """You are an helpful expert in all fields of study and the best generalist on earth who understands everything well. Use the following pieces of context from a document AND the preceding chat history to answer the user's current question.
                Provide a explanatory and elaborative answer based SOLELY on the provided context and chat history.
                If the question is a follow-up, use the chat history to understand the context of the follow-up.
                If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.
                Explain the concepts clearly and show your thinking.

                Chat History (if any):
                {chat_history}

                Retrieved Context from Document:
                {context}

                User's Current Question: {question}
                
                Elaborative Answer:"""
                CHAT_QA_PROMPT = PromptTemplate(
                    template=prompt_template_chat_qa, input_variables=["chat_history", "context", "question"]
                )
                
                try:
                    retrieved_docs = retriever.invoke(user_question) 
                    st.session_state.last_used_sources = retrieved_docs 

                    context_for_prompt = "\n\n".join([doc.page_content for doc in retrieved_docs])
                    
                    full_chat_prompt_str = CHAT_QA_PROMPT.format(
                        chat_history=history_for_prompt if history_for_prompt else "No previous chat history for this question.",
                        context=context_for_prompt,
                        question=user_question
                    )
                    
                    ai_response_text = llm_qna.invoke(full_chat_prompt_str)
                    st.session_state.chat_history.append({"role": "ai", "content": ai_response_text, "sources": retrieved_docs})
                    st.rerun()

                except Exception as e:
                    error_message = f"Error getting answer from AI: {e}"
                    st.error(error_message)
                    st.session_state.chat_history.append({"role": "ai", "content": f"Sorry, an error occurred: {e}", "sources": None})
                    st.rerun() 

    elif query_type == "Generate Practice Questions":
        st.subheader("📝 Generate Practice Questions")
        
        selected_subject_for_pq = st.selectbox(
            "Select Subject:",
            ("General", "Physics", "Chemistry", "Biology", "Geography", "History & Civics"), # You can expand this list
            key=f"subject_pq_select_{query_type_key_suffix}"
        )

        style_guidance_text = st.text_area(
            "Paste Example Questions & Answers for Style Guidance (Format: question>>answer, one per line):",
            height=200,
            key=f"pq_style_guidance_{query_type_key_suffix}",
            help="Provide 2-3 examples in the 'question>>answer' format to guide the AI's style for the selected subject. Leave blank for general style."
        )

        if st.button("Generate Questions", key=f"pq_generate_button_{query_type_key_suffix}"):
            if st.session_state.get('documents_for_direct_use'):
                with st.spinner(f"Generating {selected_subject_for_pq} practice questions..."):
                    all_doc_text = "\n".join([doc.page_content for doc in st.session_state.documents_for_direct_use])
                    document_context_for_questions = all_doc_text[:700000] 

                    questions_text = generate_practice_questions_with_guidance(
                        subject_name=selected_subject_for_pq,
                        document_text=document_context_for_questions,
                        example_qa_style_guide=style_guidance_text,
                        llm=llm_studybuddy 
                    )
                    st.markdown("### Generated Practice Questions:")
                    st.markdown(questions_text) 
            else:
                st.warning("Please upload and process a document first before generating questions.")

    elif query_type == "Generate Flashcards (Term>>Definition)":
        if st.button("Generate Flashcards", key=f"flashcard_button_{query_type_key_suffix}"):
            with st.spinner("Generating flashcards..."):
                all_doc_text = "\n".join([doc.page_content for doc in st.session_state.documents_for_direct_use])
                context_limit_flashcards = 300000 
                prompt_template_flashcards = f"""
                Based ONLY on the following text, identify key terms and their definitions.
                Format each as 'Term>>Definition'. Each flashcard should be on a new line.
                Text:
                ---
                {all_doc_text[:context_limit_flashcards]}
                ---
                Flashcards:
                """
                try:
                    response_text = llm_studybuddy.invoke(prompt_template_flashcards)
                    st.subheader("Flashcards:")
                    st.text_area("Copy these flashcards:", response_text, height=400, key=f"flashcard_output_{query_type_key_suffix}")
                except Exception as e:
                    st.error(f"Error generating flashcards: {e}")
    
    elif query_type == "Summarize Document":
        summary_session_key = f"summary_text_{query_type_key_suffix}"
        if summary_session_key not in st.session_state:
            st.session_state[summary_session_key] = ""

        summary_length = st.selectbox("Select summary length:", ("Short", "Medium", "Detailed"), key=f"summary_length_{query_type_key_suffix}")
        if st.button("Summarize", key=f"summary_button_{query_type_key_suffix}"):
            st.session_state[summary_session_key] = "" 
            with st.spinner("Summarizing..."):
                if st.session_state.get('documents_for_direct_use'):
                    all_doc_text = "\n".join([doc.page_content for doc in st.session_state.documents_for_direct_use])
                    context_limit_summary = 500000
                    length_instruction = {
                        "Short": "Provide a very brief, one-paragraph executive summary.",
                        "Medium": "Provide a multi-paragraph summary covering the main sections and key arguments.",
                        "Detailed": "Provide a comprehensive and elaborative summary, breaking down complex topics and highlighting all major sections, arguments, examples, and conclusions found in the text."
                    }
                    prompt_template_summary = f"""
                    Based ONLY on the following text, {length_instruction[summary_length]}
                    Format the output in Markdown.
                    Text:
                    ---
                    {all_doc_text[:context_limit_summary]}
                    ---
                    {summary_length} Summary (Formatted in Markdown):
                    """
                    try:
                        response_text_summary = llm_studybuddy.invoke(prompt_template_summary)
                        st.session_state[summary_session_key] = response_text_summary
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
                        st.session_state[summary_session_key] = f"Error generating summary: {e}"
                else:
                    st.warning("No document loaded to summarize.")
                    st.session_state[summary_session_key] = "No document loaded to summarize."
        
        if st.session_state.get(summary_session_key):
            st.subheader(f"{summary_length} Summary:")
            st.markdown(st.session_state[summary_session_key]) 
            st.markdown("---")
            st.text_area(
                label="Raw Markdown Summary (for copying):",
                value=st.session_state[summary_session_key] if st.session_state[summary_session_key] and "Error" not in st.session_state[summary_session_key] and "No document" not in st.session_state[summary_session_key] else "Summary not generated or error occurred.",
                height=200,
                key=f"summary_raw_text_area_{query_type_key_suffix}"
            )

elif not GEMINI_API_KEY:
    st.warning("Features are disabled as the API Key is not provided.")
else:
    st.info("👋 Upload a text-readable document in the sidebar to use the AI tools. For scanned PDFs, use the OCR section first.")

st.sidebar.markdown("---")
st.sidebar.caption("Created by Yashraj")