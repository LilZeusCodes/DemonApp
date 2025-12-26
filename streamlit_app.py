try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("Successfully switched to pysqlite3-binary.")
except ImportError:
    print("pysqlite3-binary not found, using system sqlite3. This might cause issues with ChromaDB if system sqlite3 is too old.")
except KeyError:
    print("sqlite3 module already replaced or manipulated. Assuming pysqlite3-binary is in use if installed.")

import streamlit as st
import streamlit as st
# LangChain imports for the Study Buddy section
from langchain_google_genai import GoogleGenerativeAI as LangChainGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
# Standard Python imports
import os
import tempfile
import hashlib
import time
import json # For validating/parsing JSON output from LLM

# --- OCR Specific Imports (using Gemini directly) ---
import google.generativeai as genai

# --- App Configuration & Title ---
st.set_page_config(page_title="ULTIMATE Study AI", layout="wide")
st.title("📚 YashrajAI")

# --- API Key Configuration ---
try:
    GEMINI_API_KEY = st.secrets.get("GOOGLE_API_KEY_GEMINI", os.getenv("GOOGLE_API_KEY_GEMINI"))
except (FileNotFoundError, KeyError):
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY_GEMINI")

if not GEMINI_API_KEY:
    st.error("🔴 API Key (GOOGLE_API_KEY_GEMINI) not found. Please set it. All features will be disabled.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# --- Initialize LLM and Embeddings ---
llm_studybuddy = None
llm_studybuddy2 = None
llm_qna = None
embeddings_studybuddy = None
try:
    llm_studybuddy = LangChainGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.7, google_api_key=GEMINI_API_KEY) # Lower temp for structured output
    llm_studybuddy2 = LangChainGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=1, google_api_key=GEMINI_API_KEY) 
    llm_qna = LangChainGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.7, google_api_key=GEMINI_API_KEY)
    embeddings_studybuddy = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="retrieval_document", google_api_key=GEMINI_API_KEY)
except Exception as e:
    st.sidebar.error(f"Error initializing AI models: {e}")

# --- Session State Management ---
# ... (all existing session state variables remain the same) ...
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
# --- NEW: For Mindmap ---
if 'mindmap_keywords_list' not in st.session_state:
    st.session_state.mindmap_keywords_list = ""
if 'mindmap_json_canvas' not in st.session_state:
    st.session_state.mindmap_json_canvas = ""


# =============================================
# SECTION 1: OCR PDF (Using Gemini Multimodal)
# =============================================
# ... (OCR section remains the same) ...
st.sidebar.markdown("---")
st.sidebar.header("📄 OCR Scanned PDF")
ocr_uploaded_file = st.sidebar.file_uploader("Upload a scanned PDF for OCR", type="pdf", key="gemini_ocr_uploader")

def perform_ocr_with_gemini(pdf_file_uploader_object):
    try:
        st.sidebar.write("Uploading PDF to API...")
        uploaded_gemini_file = genai.upload_file(
            path=pdf_file_uploader_object,
            display_name=pdf_file_uploader_object.name,
            mime_type=pdf_file_uploader_object.type
        )
        st.sidebar.write(f"File '{uploaded_gemini_file.display_name}' uploaded. URI: {uploaded_gemini_file.uri}. Mime Type: {pdf_file_uploader_object.type}")
        st.sidebar.write("Extracting text with AI...")
        model_ocr = genai.GenerativeModel(model_name="gemini-3-flash-preview")
        prompt = [
            "Please perform OCR on the provided PDF document and extract all text content and format it in markdown, with bold headings and leave lines wherever required.",
            "Present the extracted text clearly. If there are multiple pages, try to indicate page breaks with something like '--- Page X ---' if possible, or just provide the continuous text.",
            "Focus solely on extracting the text as accurately as possible from the document and formatting it properly.",
            uploaded_gemini_file 
        ]
        response = model_ocr.generate_content(prompt, request_options={"timeout": 600})
        try:
            genai.delete_file(uploaded_gemini_file.name)
            st.sidebar.write(f"Temporary file '{uploaded_gemini_file.display_name}' deleted from API.")
        except Exception as e_delete:
            st.sidebar.warning(f"Could not delete temporary file from API: {e_delete}")
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
        with st.spinner("Performing OCR with AI... This may take a while for large files."):
            extracted_text = perform_ocr_with_gemini(ocr_uploaded_file)
            if extracted_text:
                st.session_state.ocr_text_output = extracted_text
                st.session_state.ocr_file_name = f"ocr_of_{os.path.splitext(ocr_uploaded_file.name)[0]}.txt"
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
# ... (File upload and processing logic remains the same) ...
st.sidebar.markdown("---")
st.sidebar.header("🧠 Study AI Tools")
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
        st.sidebar.info(f"New file '{study_uploaded_file.name}' for Study AI. Processing...")
        st.session_state.vector_store = None
        st.session_state.documents_for_direct_use = None
        st.session_state.chat_history = [] 
        st.session_state.current_doc_chat_hash = current_file_hash 
        st.session_state.last_used_sources = []
        st.session_state.mindmap_keywords_list = "" # Reset mindmap data for new file
        st.session_state.mindmap_json_canvas = ""
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
                st.sidebar.error("Uploaded PDF for Study AI has no extractable text. Use OCR section first for scanned PDFs.")
                os.remove(tmp_file_path)
                st.session_state.processed_file_hash = None
            else:
                st.session_state.documents_for_direct_use = documents
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
                texts = text_splitter.split_documents(documents)
                valid_texts = [text for text in texts if text.page_content and text.page_content.strip()]
                if not valid_texts:
                    st.sidebar.error("No valid text chunks after splitting for Study Buddy.")
                else:
                    with st.spinner("Creating embeddings for Study AI..."):
                        st.session_state.vector_store = Chroma.from_documents(documents=valid_texts, embedding=embeddings_studybuddy)
                    st.session_state.processed_file_hash = current_file_hash
                    st.sidebar.success(f"✅ '{study_uploaded_file.name}' ready for Study AI!")
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
        except Exception as e:
            st.sidebar.error(f"Error processing Study AI file: {e}")
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path): os.remove(tmp_file_path)
            st.session_state.vector_store = None
            st.session_state.documents_for_direct_use = None
            st.session_state.processed_file_hash = None
            st.session_state.chat_history = []
            st.session_state.current_doc_chat_hash = None
            st.session_state.last_used_sources = []
            st.session_state.mindmap_keywords_list = ""
            st.session_state.mindmap_json_canvas = ""

# --- Backend Function for Practice Question Generation ---
# ... (generate_practice_questions_with_guidance function remains the same) ...
def generate_practice_questions_with_guidance(subject_name, document_text, example_qa_style_guide, llm):
    PRACTICE_QUESTION_PROMPT_TEMPLATE = """You are an expert AI assistant tasked with generating practice questions for a {subject_name} exam, based ONLY on the provided "Document Text". Your goal is to emulate the style, type, and difficulty of the "Example Questions and Answers" provided for style guidance.
Instructions:
1.  Carefully review the "Document Text".
2.  Carefully review the "Example Questions and Answers" to understand the desired style, question types, and answer format for {subject_name}.
3.  Generate as many new and distinct practice questions based on the "Document Text" as you can.
4.  The generated questions should be similar in nature to the provided examples.
5.  For each question you generate, provide an answer based *strictly* on the information within the "Document Text".
6.  Output Format:
    *   Each question-answer pair must be on a new line and leave a line between two pairs.
    *   Separate the question from its answer using ONLY ">>" (two greater-than signs with no spaces around them).
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
        response = llm.invoke(formatted_prompt)
        return response
    except Exception as e:
        if "response was blocked" in str(e).lower() or "safety settings" in str(e).lower():
            st.warning("The response was blocked due to safety settings. Try rephrasing style guidance or check document content.")
            return "Response blocked due to safety settings. Please check your input or document content."
        return f"Error generating practice questions: {e}"

# --- Backend Function for Custom Explanations ---
# ... (generate_custom_explanation function remains the same) ...
def generate_custom_explanation(document_text, explanation_style, llm):
    common_instructions = """
    Your goal is to explain the core concepts from the provided "Document Text" in an engaging way.
    Ensure ALL concepts from the text are covered.
    Use vivid metaphors and analogies to aid understanding.
    The explanation should be detailed and formatted in Markdown.
    Base your explanation SOLELY on the provided "Document Text".
    """
    style_specific_prompts = {
        "brainrot": f""" {common_instructions}
            Role: You are a super-online Gen Z tutor who explains things with maximum "brainrot" and internet slang, but still makes it make sense.
            Style:
            - Keep it relatively short, like a quick, punchy explainer but cover all major concepts.
            - Use current Gen Z slang, internet memes, and "brainrot" terminology (e.g., "rizz", "no cap", "it's giving...", "sus", "delulu", "based", "sigma", "gyatt" if contextually (and hilariously inappropriately) relevant, "skibidi", "fanum tax" – use these creatively and where they might (absurdly) fit an analogy).
            - Make the metaphors and analogies extremely online and relatable to internet culture.
            - It should be funny, a bit unhinged, but ultimately help someone "get" the concepts through the absurdity.
            - Don't be afraid to be a little chaotic, but ensure the core information is still conveyed.
            Document Text:
            ```
            {{document_text}}
            ```
            "Brainrot" Explanation (covering all major concepts with slang, metaphors, and analogies):
        """,
        "normal": f""" {common_instructions}
            Role: You are a clear and patient educator.
            Style:
            - The explanation should be comprehensive but concise, it should cover all major concepts in an easily digestable format.
            - Use clear, easy-to-understand language.
            - Employ insightful metaphors and analogies to clarify complex points.
            - Maintain a helpful and encouraging tone.
            Document Text:
            ```
            {{document_text}}
            ```
            Normal Explanation (covering all major concepts with clear metaphors and analogies):
        """
    }
    selected_prompt_template = style_specific_prompts.get(explanation_style.lower(), style_specific_prompts["normal"])
    formatted_prompt = selected_prompt_template.format(document_text=document_text)
    try:
        response = llm.invoke(formatted_prompt)
        return response
    except Exception as e:
        if "response was blocked" in str(e).lower() or "safety settings" in str(e).lower():
            st.warning("The explanation response was blocked due to safety settings. The document content might be triggering filters.")
            return "Response blocked due to safety settings. Please check the document content."
        return f"Error generating explanation: {e}"

# --- NEW: Backend Functions for Mindmap Generation ---
def extract_keywords_for_mindmap(document_text, llm):
    """Extracts keywords and a central topic for mindmap generation."""
    prompt = f"""
    Based on the following Document Text, identify the main central topic and up to 10-15 key related concepts, terms, or sub-topics.
    The goal is to gather elements for creating a mind map.
    List the central topic first, then the related keywords.
    Format the output as:
    Central Topic: [Your identified central topic]
    Keywords:
    - Keyword 1
    - Keyword 2
    - Keyword 3
    ...

    Document Text:
    ```
    {document_text}
    ```

    Extracted Central Topic and Keywords:
    """
    try:
        response = llm.invoke(prompt)
        # Basic parsing (can be made more robust)
        lines = response.strip().split('\n')
        central_topic = "Unknown Central Topic"
        keywords = []
        parsing_keywords = False
        for line in lines:
            if line.lower().startswith("central topic:"):
                central_topic = line.split(":", 1)[1].strip()
            elif line.lower().startswith("keywords:"):
                parsing_keywords = True
            elif parsing_keywords and line.startswith("- "):
                keywords.append(line[2:].strip())
        if not keywords and central_topic == "Unknown Central Topic" and response.strip(): # Fallback if parsing fails
            return "Central Topic: Document Analysis\nKeywords:\n- " + "\n- ".join(response.strip().split('\n')[:10]), central_topic, response.strip().split('\n')[:10]


        return response, central_topic, keywords # Return raw response, parsed central_topic, and parsed keywords
    except Exception as e:
        return f"Error extracting keywords: {e}", None, []


def generate_json_canvas_from_keywords(central_topic, keywords, llm, example_canvas_json_str):
    """Generates JSON Canvas from keywords, guided by an example structure."""
    # Create a simplified list of keywords for the prompt to keep it manageable
    keyword_list_for_prompt = "- " + "\n- ".join(keywords) # Use up to 10 keywords for canvas generation

    prompt = f"""
    You are an AI assistant that helps create mind maps in JSON Canvas format.
    Your task is to take a Central Topic and a list of Related Keywords and structure them into a valid JSON Canvas object containing 'nodes' and 'edges'.

    **Instructions for JSON Canvas Output:**
    1.  The mindmap should be interconnected with small explanations of relationships between the connected keywords. the mindmap should resemble how neurons are connected in the brain. Group related and unconnected nodes into a list
    2.  This JSON object MUST contain two top-level keys: "nodes" (an array of node objects) and "edges" (an array of edge objects).
    3.  **Nodes:**
        *   Each node object must have:
            *   `"id"`: A unique string identifier (e.g., "node_topic", "node_keyword1"). Make them descriptive based on the text.
            *   `"x"`, `"y"`: Integer coordinates. Try to arrange keywords around the central topic. For example, central topic at (0,0), keywords at (200,0), (-200,0), (0,200), (0,-200), (150,150) etc. Vary these.
            *   `"width"`: Integer (e.g., 250).
            *   `"height"`: Integer (e.g., 60).
            *   `"type"`: Always set to `"text"`.
            *   `"text"`: The actual keyword or central topic string.
    4.  **Edges:**
        *   Each edge object must have:
            *   `"id"`: A unique string identifier (e.g., "edge_topic_keyword1").
            *   `"fromNode"`: The `id` of the starting node (usually a keyword node or the central topic).
            *   `"toNode"`: The `id` of the ending node (usually the central topic if connecting keywords to it, or another keyword).
            *   `"label"` (optional): A short string describing the relationship (e.g., "relates to", "is part of", "leads to"). If unsure, omit or use a generic label.
        *   Ensure all keywords are connected to the central topic, or form a logical structure.
    5.  Use EVERY SINGLE keyword provided in the Related Keywords list.
    **Example of the desired JSON Canvas structure (DO NOT just copy this, adapt it based on the provided keywords):**
    ```json
    {example_canvas_json_str}
    ```

    **Now, generate the JSON Canvas for the following:**

    Central Topic: {central_topic}

    Related Keywords:
    {keyword_list_for_prompt}

    Strictly adhere to the JSON format. Ensure all quotes are correctly escaped if necessary within text fields (though unlikely for keywords).
    The output should be ONLY the JSON object, starting with `{{` and ending with `}}`.
    ---
    JSON Canvas Output:
    """
    try:
        response = llm.invoke(prompt)
        # Attempt to clean and validate the JSON
        # LLMs can sometimes add ```json ... ``` markdown, so strip it
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"): # General ```
             cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        
        # Validate JSON
        json.loads(cleaned_response) # This will raise an error if not valid JSON
        return cleaned_response
    except json.JSONDecodeError as e_json:
        st.error(f"LLM returned invalid JSON for mindmap: {e_json}")
        st.text_area("Problematic LLM Output (JSON Canvas):", cleaned_response, height=200)
        return f'{{"error": "Failed to generate valid JSON for mindmap. LLM output was not valid JSON.", "details": "{str(e_json).replace("\"", "'")}", "raw_output": "{cleaned_response.replace("\"", "'")}"}}' # Return error JSON
    except Exception as e:
        return f'{{"error": "Error generating JSON canvas: {str(e).replace("\"", "'")}"}}'


# --- Main Interaction Area for Study Buddy Tools ---
if st.session_state.get('vector_store') and st.session_state.get('documents_for_direct_use') and GEMINI_API_KEY and llm_qna and llm_studybuddy:
    st.markdown("---")
    if st.session_state.current_doc_chat_hash != st.session_state.processed_file_hash:
        st.session_state.chat_history = []
        st.session_state.current_doc_chat_hash = st.session_state.processed_file_hash
        st.session_state.last_used_sources = []
        st.session_state.mindmap_keywords_list = ""
        st.session_state.mindmap_json_canvas = ""
        
    header_file_name = "your document"
    if study_uploaded_file and hasattr(study_uploaded_file, 'name'):
        if st.session_state.processed_file_hash == hashlib.md5(study_uploaded_file.getvalue()).hexdigest():
            header_file_name = study_uploaded_file.name
            
    st.header(f"🛠️ Study Tools for: {header_file_name}")
    
    query_type_key_suffix = st.session_state.processed_file_hash or "default_study_tools"
    
    tool_options = ["Chat & Ask Questions", 
                    "Generate Practice Questions",
                    "Create Explanation",
                    "Create Keywords Mindmap", # NEW OPTION
                    "Generate Flashcards (Term>>Definition)", 
                    "Summarize Document"]
    query_type = st.radio(
        "What do you want to do with the text-readable document?",
        tool_options,
        key=f"query_type_{query_type_key_suffix}"
    )

    # ... (Chat & Ask Questions, Practice Questions, Create Explanation, Flashcards, Summarize sections remain the same) ...
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
                
                Elaborative Answer:""" # Your updated prompt
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
            ("General", "Physics", "Chemistry", "Biology", "Geography", "History & Civics"),
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
    elif query_type == "Create Explanation":
        st.subheader("💡 Create Custom Explanation")
        explanation_style_selected = st.selectbox(
            "Select Explanation Style:",
            ("Normal", "Brainrot"),
            key=f"exp_style_select_{query_type_key_suffix}"
        )
        if st.button("Generate Explanation", key=f"exp_generate_button_{query_type_key_suffix}"):
            if st.session_state.get('documents_for_direct_use'):
                with st.spinner(f"Generating '{explanation_style_selected}' style explanation..."):
                    all_doc_text = "\n".join([doc.page_content for doc in st.session_state.documents_for_direct_use])
                    document_context_for_explanation = all_doc_text[:700000] 
                    explanation_text = generate_custom_explanation(
                        document_text=document_context_for_explanation,
                        explanation_style=explanation_style_selected,
                        llm=llm_studybuddy2
                    )
                    st.markdown(f"### {explanation_style_selected} Explanation:")
                    st.markdown(explanation_text)
            else:
                st.warning("Please upload and process a document first before generating an explanation.")
    
    # --- NEW: Create Keywords Mindmap Section ---
    elif query_type == "Create Keywords Mindmap":
        st.subheader("🗺️ Create Keywords Mindmap")

        if st.button("Generate Keywords & Mindmap", key=f"mindmap_generate_button_{query_type_key_suffix}"):
            st.session_state.mindmap_keywords_list = "" # Clear previous
            st.session_state.mindmap_json_canvas = ""   # Clear previous

            if st.session_state.get('documents_for_direct_use'):
                with st.spinner("Step 1: Extracting keywords..."):
                    all_doc_text = "\n".join([doc.page_content for doc in st.session_state.documents_for_direct_use])
                    document_context_for_mindmap = all_doc_text[:500000] # Adjust context as needed

                    raw_keywords_output, central_topic, keywords = extract_keywords_for_mindmap(
                        document_text=document_context_for_mindmap,
                        llm=llm_studybuddy
                    )
                    st.session_state.mindmap_keywords_list = raw_keywords_output # Store the raw list with central topic

                if central_topic and keywords:
                    st.markdown("### Extracted Keywords:")
                    st.markdown(st.session_state.mindmap_keywords_list) # Show the extracted list
                    st.markdown("---")

                    with st.spinner("Step 2: Generating JSON Canvas Mindmap... (This can be slow and experimental)"):
                        # Provide the example JSON canvas string directly in the call
                        example_canvas_str = """
                        {
                        "nodes": [
                        {"id":"node_photosynthesis_center","x":-50,"y":-150,"width":280,"height":60,"type":"text","text":"Photosynthesis (Central Topic)"},
                        {"id":"node_sunlight","x":-350,"y":-300,"width":200,"height":50,"type":"text","text":"Sunlight (Input)"},
                        {"id":"node_water","x":-350,"y":-50,"width":200,"height":50,"type":"text","text":"Water (Input)"},
                        {"id":"node_co2","x":250,"y":-300,"width":250,"height":50,"type":"text","text":"Carbon Dioxide (Input)"},
                        {"id":"node_chlorophyll","x":-50,"y":-20,"width":250,"height":50,"type":"text","text":"Chlorophyll (Catalyst/Location)"},
                        {"id":"node_glucose","x":-350,"y":100,"width":200,"height":50,"type":"text","text":"Glucose (Output)"},
                        {"id":"node_oxygen","x":250,"y":100,"width":200,"height":50,"type":"text","text":"Oxygen (Output)"}
                        ],
                        "edges": [
                        {"id":"edge_sun_photo","fromNode":"node_sunlight","toNode":"node_photosynthesis_center","label":"is required for"},
                        {"id":"edge_water_photo","fromNode":"node_water","toNode":"node_photosynthesis_center","label":"is required for"},
                        {"id":"edge_co2_photo","fromNode":"node_co2","toNode":"node_photosynthesis_center","label":"is required for"},
                        {"id":"edge_chloro_photo","fromNode":"node_chlorophyll","toNode":"node_photosynthesis_center","label":"is site of / uses"},
                        {"id":"edge_photo_glucose","fromNode":"node_photosynthesis_center","toNode":"node_glucose","label":"produces"},
                        {"id":"edge_photo_oxygen","fromNode":"node_photosynthesis_center","toNode":"node_oxygen","label":"produces"}
                        ]
                        }                        """
                        json_canvas_output = generate_json_canvas_from_keywords(
                            central_topic=central_topic,
                            keywords=keywords,
                            llm=llm_studybuddy2, # Use LLM with Higher temperature
                            example_canvas_json_str=example_canvas_str
                        )
                        st.session_state.mindmap_json_canvas = json_canvas_output
                elif raw_keywords_output and "Error extracting keywords:" in raw_keywords_output:
                    st.error(raw_keywords_output) # Show error from keyword extraction
                else:
                    st.warning("Could not extract enough information to generate a mindmap.")
            else:
                st.warning("Please upload and process a document first.")

        if st.session_state.mindmap_keywords_list and not st.session_state.mindmap_json_canvas : # If keywords were extracted but mindmap not yet shown (e.g. after button click)
            if "Error extracting keywords:" not in st.session_state.mindmap_keywords_list and \
               "Unknown Central Topic" not in st.session_state.mindmap_keywords_list : # Avoid showing if only error
                st.markdown("### Extracted Keywords:")
                st.markdown(st.session_state.mindmap_keywords_list)
                st.markdown("---")


        if st.session_state.mindmap_json_canvas:
            st.markdown("### JSON Canvas Mindmap Data:")
            
            # Attempt to pretty-print if it's valid JSON, otherwise show raw
            try:
                parsed_json = json.loads(st.session_state.mindmap_json_canvas)
                pretty_json_canvas = json.dumps(parsed_json, indent=2)
                is_valid_json_for_download = True
            except json.JSONDecodeError:
                pretty_json_canvas = st.session_state.mindmap_json_canvas # Show raw if not valid
                is_valid_json_for_download = False
                st.warning("The generated mindmap data is not valid JSON. Download might not work as expected in canvas tools.")

            st.text_area("JSON Canvas Output (for copying or inspection):", pretty_json_canvas, height=300, key=f"mindmap_json_output_{query_type_key_suffix}")
            
            if is_valid_json_for_download:
                mindmap_file_name = f"mindmap_{header_file_name.replace(' ', '_').split('.')[0]}.canvas"
                st.download_button(
                    label="📥 Download Mindmap (.canvas file)",
                    data=pretty_json_canvas.encode('utf-8'),
                    file_name=mindmap_file_name,
                    mime="application/json", # .canvas files are essentially JSON
                    key=f"download_mindmap_{query_type_key_suffix}"
                )
            st.info("You can import this .canvas file into Obsidian or other compatible JSON Canvas tools.")
    # --- End NEW Section ---
    
    elif query_type == "Generate Flashcards (Term>>Definition)":
        # ... (Flashcard logic remains the same) ...
        if st.button("Generate Flashcards", key=f"flashcard_button_{query_type_key_suffix}"):
            with st.spinner("Generating flashcards..."):
                all_doc_text = "\n".join([doc.page_content for doc in st.session_state.documents_for_direct_use])
                context_limit_flashcards = 300000 
                prompt_template_flashcards = f"""
                Based ONLY on the following text, identify key words and their meanings.
                Format each as 'Word>>Meaning'. Each flashcard should be on a new line.
                Examples:
                - 'Photosynthesis>>The process by which green plants use sunlight to synthesize foods with the help of chlorophyll.'
                - 'Mitosis>>A type of cell division that results in two daughter cells each having the same number and kind of chromosomes as the parent nucleus.'
                - 'Oblivious>>Unaware or unconcerned about what is happening around one.'
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
        # ... (Summarize Document logic remains the same) ...
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
                        "Detailed": "Provide an elaborative summary, breaking down complex topics and highlighting all major sections, arguments, examples, and conclusions found in the text. Go over ALL concepts and ideas presented in the text"
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
            current_summary_text = st.session_state[summary_session_key]
            st.markdown(current_summary_text)
            
            if current_summary_text and "Error generating summary" not in current_summary_text and "No document loaded" not in current_summary_text:
                st.markdown("---")
                summary_file_name = f"summary_{header_file_name.replace(' ', '_').split('.')[0]}_{summary_length.lower()}.md"
                st.download_button(
                    label="📥 Download Summary (Markdown)",
                    data=current_summary_text.encode('utf-8'),
                    file_name=summary_file_name,
                    mime="text/markdown",
                    key=f"download_summary_{query_type_key_suffix}"
                )
            st.markdown("---")
            st.text_area(
                label="Raw Markdown Summary (for copying):",
                value=current_summary_text if current_summary_text and "Error" not in current_summary_text and "No document" not in current_summary_text else "Summary not generated or error occurred.",
                height=200,
                key=f"summary_raw_text_area_{query_type_key_suffix}"
            )


elif not GEMINI_API_KEY:
    st.warning("AI features are disabled as the API Key is not provided.")
else:
    st.info("👋 Upload a text-readable document in the sidebar to use the Study AI tools. For scanned PDFs, use the OCR section first.")

st.sidebar.markdown("---")
st.sidebar.caption("Created by Yashraj.")
