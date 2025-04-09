import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
#step LLMh
hf_token = os.getenv("hf_token")
huggingface_repo_id="mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm=HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,

        task='text-generation',
        model_kwargs={"token": hf_token,
                      "max_length":1024}
    )
    return llm

DB_FAISS_PATH = "vector/db_faiss"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt_template(template_str):
    prompt = PromptTemplate(
        template=template_str,
        input_variables=["context", "question"]
    )
    return prompt

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH,embeddings_model, allow_dangerous_deserialization=True)


#create qa chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(huggingface_repo_id),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt_template(CUSTOM_PROMPT_TEMPLATE)}
)
# now innvoke with a single query
user_query=input("write query  here: ")
response=qa_chain.invoke({'query':user_query})
print(response['result'])

