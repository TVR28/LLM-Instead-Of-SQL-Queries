from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import streamlit as st
from streamlit_option_menu import option_menu
from pandasai.connectors import MySQLConnector
from pandasai import SmartDataframe
from PIL import Image
import io
import os

load_dotenv()

llm = ChatOpenAI(model="gpt-4")

def init_database() -> SQLDatabase:
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    database = os.getenv("DB_NAME")
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
    template = """
    You are a Pro Assist, an AI analyst at a company built to answer queries of users. You are interacting with a user who is asking you questions about the comapany's data.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: How many symbols are there?
    SQL Query: SELECT COUNT(DISTINCT symbol) AS num_symbols FROM ivindata;
    Question: Name top 10 candidate names.
    SQL Query: SELECT candidate_name FROM ivindata ORDER BY `general` DESC LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOpenAI(model="gpt-4")
    # llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template = """
    You are Pro Assist, an AI analyst at a company built to answer queries of users. You are interacting with a user who is asking you questions about the company's data.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatOpenAI(model="gpt-4")
    # llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })


st.set_page_config(page_title="ProAssist", page_icon=":speech_balloon:")


with st.sidebar:
    selected = option_menu(
        menu_title="Pro Assist",
        options = ["Chat", "Visualize"],
        
    )

if selected == "Chat":
    st.title("Pro Assist")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm Pro Assist! Ask me anything about 2024 Elections."),
        ]
    
    # Initialize the database connection once and store it in session state
    if "db" not in st.session_state:
        with st.spinner("Connecting to database..."):
            db = init_database()
            st.session_state.db = db
            st.success("Connected to database!")
            
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
                
    user_query = st.chat_input("Type a message...")
    if user_query is not None and user_query.strip() != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        with st.chat_message("Human"):
            st.markdown(user_query)
            
        with st.chat_message("AI"):
            # To check SQL Query
            # sql_chain = get_sql_chain(st.session_state.db)
            # response = sql_chain.invoke({
            #     "chat_history": st.session_state.chat_history,
            #     "question": user_query
            # })
            
            # To check Natural Language Response From LLM
            response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
            st.markdown(response)
            
        st.session_state.chat_history.append(AIMessage(content=response))
        
if selected == "Visualize":
    
    st.title("ProAssist Visualize")
    
    sql_connector = MySQLConnector(
        config={
            "host" : os.getenv("DB_HOST"),
            "port" : os.getenv("DB_PORT"),
            "database" : os.getenv("DB_NAME"),
            "username" : os.getenv("DB_USER"),
            "password" : os.getenv("DB_PASSWORD"),
            "table" : os.getenv("DB_TABLE"),
        }
    )
    
    df_connector = SmartDataframe(sql_connector, config ={"llm": llm})
    
    user_query = st.text_input("Ask Me Anything")
    
    if st.button("Generate"):
        if user_query:
            with st.spinner("Generating Response..."):
                st.write(df_connector.chat(user_query))
