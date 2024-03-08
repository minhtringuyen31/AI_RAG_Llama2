from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "models/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

def createChain(llm, db);
    llm_chain = ConversationalRetrievalChain(
        llm = llm, 
        retriever = db.as_retriever()
    )
    return llm_chain