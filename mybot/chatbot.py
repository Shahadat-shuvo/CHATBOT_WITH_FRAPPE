from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory, ConversationSummaryMemory
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
import os

os.environ["OPENAI_API_KEY"] = "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"

loaders = [PyPDFLoader("/home/friday/Frappe/frappe-bench/sites/shuvo.com/public/files/intern.pdf"),
            PyPDFLoader("/home/friday/Frappe/frappe-bench/sites/shuvo.com/public/files/constitution.pdf")]

customs_template = """Use the following pieces of context to answer the question at the end. \n
If you don't know the answer, just say that you don't know, don't try to make up an answer. \n
If the question is irrelevant to the context, just tell "Please! make your question with relevant info". 
 Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
 use previous chat history to make answer.
current conversation: 
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(customs_template)

prompt = PromptTemplate(
    input_variables=["question","context" ], template=customs_template
)

documents = []
for loader in loaders:
    documents.extend(loader.load())

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
        memory1 = ConversationSummaryMemory(
                                            llm=OpenAI(temperature=0),
                                            # chat_memory=ChatMessageHistory(),
                                            # return_messages=True
                                        )
        # define retriever
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        # qa_chain = RetrievalQA.from_chain_type(
        #             llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        #             retriever=retriever,
        #             # memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
        #             return_source_documents=True,
        #             chain_type_kwargs={"prompt": prompt}
        #         )

        qa = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
                chain_type="stuff",
                retriever=retriever,
                # condense_question_prompt=promptss,
                return_source_documents=True,
                return_generated_question=True,
            )
        result = qa({"question": user_input, "chat_history": self.chat_history})
        # result = qa_chain({"query": user_input, "chat_history": self.chat_history})
        self.chat_history.extend([(user_input, result["answer"])])
        response = result['answer']
        # response1 = result['chat_history']
        print(response)
        # print(self.chat_history)
        # print(result["generated_question"])
        # print(result["source_documents"])
        return response


# chats = getChat()
chats2 = getChat2()
while True:
    user_input = input("> ")
    chats2.get_response(user_input)

