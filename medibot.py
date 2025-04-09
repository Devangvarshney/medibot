import os
import gradio as gr

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Path to your FAISS vector database
DB_FAISS_PATH = "vectorstore/db_faiss"

# Load HuggingFace token from environment variable
hf_token = os.getenv("hf_token")

# Load vectorstore
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Prompt Template
def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# Load LLM from HuggingFace
def load_llm(huggingface_repo_id, hf_token):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        max_length=512,
        huggingfacehub_api_token=hf_token
    )

# Custom Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don’t know the answer, just say that you don’t know — don’t try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

# Load everything once
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = load_llm("mistralai/Mistral-7B-Instruct-v0.3", hf_token)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Chatbot function
def chat(prompt, history):
    try:
        response = qa_chain.invoke({'query': prompt})
        answer = response['result']
        history.append((prompt, answer))
        return history, history
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        history.append((prompt, error_msg))
        return history, history

# Description
description = (
    "**🧠 Chat with a futuristic AI assistant trained on medical knowledge.**\n\n"
    "**📚 Reference Books Used:**\n"
    "- *Atlas of Human Anatomy – Frank H. Netter, M.D.*\n"
    "- *Guyton and Hall Medical Physiology*\n"
    "- *Park's Preventive and Social Medicine*\n"
    "- *Gale Encyclopedia of Medicine*"
)

# Gradio UI
with gr.Blocks(css="""
body {background: #0f0f0f; font-family: 'Segoe UI', sans-serif;}
.gradio-container {padding: 30px;}
.markdown {color: #00ffcc;}
textarea, .gr-input, .gr-button {border-radius: 12px !important;}
.gr-button {transition: all 0.3s ease-in-out;}
.gr-button:hover {box-shadow: 0 0 10px #00ffcc;}
.gr-chatbot {background: rgba(255,255,255,0.03); border-radius: 12px;}
#chatbox {
    height: 300px !important;
    font-size: 18px !important;
    overflow-y: auto !important;
}
""") as demo:
    # Header
    gr.Markdown(f"## 🤖 **Ask Your Medical AI Assistant**\n\n{description}")

    state = gr.State([])

    with gr.Row():
        user_input = gr.Textbox(placeholder="Ask your question here...", lines=1, label="💬 Your Question")

    # Chat History with custom height and font
    chat_output = gr.Chatbot(label="🧾 Chat History", elem_id="chatbox")

    with gr.Row():
        submit_btn = gr.Button("🚀 Submit", variant="primary")
        clear_btn = gr.Button("🧹 Clear Chat", variant="secondary")
        flag_btn = gr.Button("⚠️ Flag", variant="secondary")

    submit_btn.click(fn=chat, inputs=[user_input, state], outputs=[chat_output, state])
    user_input.submit(fn=chat, inputs=[user_input, state], outputs=[chat_output, state])

    clear_btn.click(fn=lambda: ([], []), inputs=[], outputs=[chat_output, state])
    flag_btn.click(fn=lambda: ([], []), inputs=[], outputs=[chat_output, state])  # Placeholder

    # Floating Voice & Vision Bot Button - Top Right Corner
    gr.HTML("""
        <div style="position: fixed; top: 20px; right: 20px; z-index: 1000;">
            <a href="https://voice-and-vision-bot2-1.onrender.com" target="_blank" 
               style="background-color: #4CAF50; color: white; padding: 12px 20px; text-align: center;
                      text-decoration: none; display: inline-block; font-size: 14px; border-radius: 10px;
                      box-shadow: 0px 2px 5px rgba(0,0,0,0.3);">
                🔗 Open Voice & Vision Bot
            </a>
        </div>
    """)

if __name__ == "__main__":
    demo.launch()
