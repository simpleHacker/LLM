class Rag:
    VECTOR_DB_PATH = "vector_dbs"
    TEXT_VECTOR_DB_NAME = "text_vector.db"
    IMAGE_VECTOR_DB_NAME = "image_vector.db"
    TEXT_VECTOR_DB_STORE = f"./{VECTOR_DB_PATH}/{TEXT_VECTOR_DB_NAME}"
    #EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    MODEL_PATH = "models"
    EMBEDDING_MODEL = f"./{MODEL_PATH}/all-mpnet-base-v2"
  
    def __init__(self):
      self.__text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20, length_function=len,)
      self.__image_text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10, separator = "\n",)
      self.__session_id = ""
      self.__chat_history = {}
      self.__session_store = {}
      self.__text_db = None
      self.__image_db = None
      self.__ai_client = None
      self.__text_documents = []
      self.__images_documents = []
      self.__text_retriever = None

    def prepare(self):
      embedding = self.get_embeddings()
      self.__text_db = self.create_text_vector_db(embeddings)
      self.__ai_client = self.create_ai_client()
        
  
    def create_ai_client(self):
      client = ChatOpenAI(
        model="<model name>",
        api_key="<auth_token>",
        base_url="<end point>",
        timeout=httpx.Timeout(60.0),
        temperature=0.1,
        max_retries=3,
        max_tokens=5000
      )
      return client

    def get_embeddings():
      model_kwargs={"device" : "cpu"}
      encode_kwargs={'normalize_embeddings' : True}
      embeddings = HuggingFaceEmbeddings(
        model_name="<model_path or name>",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
      )

    def load_texts(path, reload=False):
      if not reload and os.path.exists(self.TEXT_VECTOR_DB_STORE):
        return
      for filename in os.listdir(path):
        f = os.path.join(path, filename)
        documents = TextLoader(f).load()
        self.__text_documents.extend(self.__text_splitter.split_documents(documents))

    def create_text_vector_db(embeddings, docs):
      if os.path.exists(self.TEXT_VECTOR_DB_STORE):
        db = FAISS.load_local(self.TEXT_VECTOR_DB_STORE, embeddings=embeddings, allow_dangerous_deserialization=True)
      else:
        db = FAISS.from_documents(self.__text_documents, embeddings)
        db.save_local(self.TEXT_VECTOR_DB_STORE)
        self.__s3_sync.upload_vector_db("text_vector.db")
      return db.as_retriever()

    def set_session_id(self, session_id):
      self.__session_id = session_id

    def get_session_history(self)  -> BaseChatMessageHistory:
      if self.__session_id not in self.__session_store:
        self.__session_store[self.__session_id] = ChatMessageHistory()
      return self.__session_store[self.__session_id]

    def get_qa_prompt(self):
      qa_system_prompt = """ Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Use three sentences maximum and keep the answer concise.\
        {context}"""
      qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        return qa_prompt

    def get_context_prompt(self):
      contextualize_q_system_prompt="""Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        return contextualize_q_prompt

    def create_text_retriever(self):
      history_retriever = create_history_aware_retriever(self.__ai_client, self.__text_db, self.get_context_prompt())
      question_answer_chain = crate_stuff_documents_chain(self.__ai_client, self.get_qa_prompt())
      chain = create_retrieval_chain(history_retriever, question_answer_chain)
      self.__text_retriever = RunnableWithMessageHistory(chain, self.get_session_history, input_messages_key="input", history_messages_key="chat_history", output_messages_key="answer")
      return self.__text_retriever

    def retrieveText(self, query):
        if query == "reset":
            self.__chat_history[self.__session_id] = [] # reset chat history
            return "Start a new topic :)"
        if self.__session_id not in self.__chat_history:
            self.__chat_history[self.__session_id] = []
        response = self.__text_retriever.invoke(
            {"input": query,
             "chat_history": self.__chat_history[self.__session_id]},
             config = { "configurable": {"session_id": self.__session_id}})
        self.__chat_history[self.__session_id].extend([HumanMessage(content=query), response["answer"]])
        source = response["context"][0].metadata["source"]
        answer = response["answer"]
        return answer, source
        # return response["answer"]

"""
session_id = "demo1"
rag = Rag()
rag.load_texts("<text_docs_path>")
rag.prepare(True)
rag.set_sessionPid(session_id)
rag.create_text_retriever()

a = rag.retrieveText(query)
"""
