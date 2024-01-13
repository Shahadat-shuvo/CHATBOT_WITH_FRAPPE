from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import RedisChatMessageHistory, ConversationBufferMemory
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import PromptTemplate


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA,  ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader
import param
import frappe

# # from langchain.prompts import (
# #     ChatPromptTemplate,
# #     HumanMessagePromptTemplate,
# #     MessagesPlaceholder,
# # )
# # from langchain.schema import SystemMessage
import os


# promptss = ChatPromptTemplate.from_messages(
#     [
#         SystemMessage(
#             content="You are a chatbot having a conversation with a human."
#         ),  # The persistent system prompt
#         MessagesPlaceholder(
#             variable_name="chat_history"
#         ),  # Where the memory will be stored.
#         HumanMessagePromptTemplate.from_template(
#             "{user_input}"
#         ),  # Where the human input will injected
#     ]
# )

tuition_template = """
    I want you as my assistance. Your primary task is to generate response according to my questions. /
    This is a dialogue between us. Don't act as your own. Make a Sorry response If the answer is not there. Make the response short and informative. 
    Follow Up Input: {question}
    Standalone question:
"""

custom_template = """Use the following pieces of context to answer the question at the end. Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language. If you do not know the answer reply with 'I am sorry'.
                        Make the response short. Do not make any extra response.

{context}
Follow Up Input: {question}
Your response:
"""

customs_template = """Use the following pieces of context to answer the question at the end. \n
If you don't know the answer, just say that you don't know, don't try to make up an answer. \n
If the question is irrelevant to the context, just tell "Please! make your question with relevant info". 
 Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
 use previous chat history to make answer.
current conversation: {chat_history}
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(customs_template)

prompt = PromptTemplate(
    input_variables=["chat_history","question", "context"], template=customs_template
)

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)
# prompts = CUSTOM_QUESTION_PROMPT.format(chat_history="chat_history", question="question")
# prompt = PromptTemplate(
#     input_variables=["question"],
#     template=custom_template
# )

os.environ["OPENAI_API_KEY"] = "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"

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key="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")

loaders = [PyPDFLoader("/home/friday/Frappe/frappe-bench/sites/shuvo.com/public/files/intern.pdf"),
            PyPDFLoader("/home/friday/Frappe/frappe-bench/sites/shuvo.com/public/files/constitution.pdf")]

documents = []
for loader in loaders:
    documents.extend(loader.load())

# documents = loader.load()

chat_history = []


class getChat():
    def get_response(self, user_input):
        # response = llm(user_input)
        # response = prompt.format(user_question=user_input)
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(user_input)
        return response
    
    def get_chat(self):
        # chat_history = [(query, result["answer"])]
        if not chat_history:
            return f'No history Yet'
        rlist = chat_history
        for exchange in chat_history:
            rlist.append(exchange)
        print(rlist)
        return rlist

class getChat2(param.Parameterized):
    chat_history = param.List([])

    def get_response(self, user_input):
         # split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=30)
        docs = text_splitter.split_documents(documents)

        # define embedding
        embeddings = OpenAIEmbeddings()

        # create vector database from data
        db = DocArrayInMemorySearch.from_documents(docs, embeddings)

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # define retriever
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # qa_chain = RetrievalQA.from_chain_type(
        #             llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        #             retriever=retriever,
        #             memory=memory,
        #             # return_source_documents=True,
        #             chain_type_kwargs={"prompt": prompt}
        #         )

        qa = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
                # chain_type="stuff",
                retriever=retriever,
                memory=memory,
                # condense_question_prompt=promptss,
                # return_source_documents=True,
                # return_generated_question=True,
            )
        result = qa({"question": user_input, "chat_history": self.chat_history})
        # result = qa_chain({"query": user_input, "chat_history": chat_history})
        self.chat_history.extend([(user_input, result["answer"])])
        response = result['answer']
        response1 = result['chat_history']
        print(response)
        print(self.chat_history)
        # print(result["generated_question"])
        # print(result["source_documents"])
        return response
    
    def get_chats(self):
       if not self.chat_history:
           return f"This is empty"
       rlist= []
       for exchange in self.chat_history:
           rlist.append(str(exchange))
       return rlist
    
    def clr_history(self):
        self.chat_history = []
        return self.chat_history

# chats = getChat()
# chats2 = getChat2()
# while True:
#     user_input = input("> ")
#     chats2.get_response(user_input)
@frappe.whitelist()
def get_chat_response(user_input):
    response = chats2.get_response(user_input)
    return response 


@frappe.whitelist()
def get_chat_history():
     getChats = chats2.get_chats()
     return getChats

# @frappe.whitelist()
# def clear_history():
#     clrHist = chats.clr_history()
#     return clrHist

import frappe 

# create a new document
