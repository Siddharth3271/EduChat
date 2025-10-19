import os,re
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from flask import Flask, render_template, request, redirect
from PyPDF2 import PdfReader

import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

API_KEY = os.getenv("GOOGLE_API_KEY") 
# print("API Key:", os.getenv("GOOGLE_API_KEY"))

start_greeting=["hi","hello"]
end_greeting=["bye"]
way_greeting=["who are you"]

#Using this folder to store the data from the frontend
DATA_DIR="__data__"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

#flask app
app=Flask(__name__)
vectorstore=None
conversation_chain=None
chat_history=[]
rubric_text=""
# googleai_client=ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY)

class HumanMessage:
    def __init__(self,content):
        self.content=content

    def __repr__(self):
        return f'HumanMessage(content={self.content})'
    

class AIMessage:
    def __init__(self,content):
        self.content=content

    def __repr__(self):
        return f'AIMessage(content={self.content})'
    
#pdf extraction
def get_pdf_text(pdf_docs):
    text=""
    pdf_text=""
    for pdf in pdf_docs:
        filename=os.path.join(DATA_DIR,pdf.filename)
        pdf_text=""
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
            pdf_text+=page.extract_text()

        with (open(filename,"w",encoding="utf-8")) as op_file:
            op_file.write(pdf_text)
    
    return text

#splitting the text in sub parts
def get_text_chunks(text):
    text_splitter=CharacterTextSplitter(separator="\n",chunk_size=1000,chunk_overlap=200,length_function=len)

    chunks=text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(google_api_key=API_KEY,model="models/gemini-embedding-001")
    vectorstore=FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm=ChatGoogleGenerativeAI(google_api_key=API_KEY, model="gemini-2.5-flash")
    vc=vectorstore.as_retriever()
    memory=ConversationBufferMemory(
        memory_key='chat_history',return_messages=True
    )
    conversation_chain=ConversationalRetrievalChain.from_llm(llm=llm,retriever=vc,memory=memory)

    return conversation_chain

def _grade_essay(essay, rubric_text):
    """Grades an essay using the Google Generative AI model via LangChain."""
    try:
        # FIX: Use the ChatGoogleGenerativeAI wrapper consistently
        llm = ChatGoogleGenerativeAI(google_api_key=API_KEY, model="gemini-2.5-flash", temperature=0.4)
        
        # Define the system prompt template (the rubric instruction)
        template = """
        You are an expert academic essay grader and writing coach.
        Analyze the following essay based ONLY on the criteria in the Rubric section.
        Your output MUST be a score summary (out of 100), detailed feedback by criterion, and improvement suggestions.

        --- RUBRIC ---
        {rubric}

        --- ESSAY TO GRADE ---
        {essay}
        """
        
        # Create a prompt value and invoke the model
        essay_prompt = PromptTemplate.from_template(template).format(rubric=rubric_text, essay=essay)
        
        # Invoke the LLM using the LangChain method
        response = llm.invoke(essay_prompt)

        # Convert line breaks for HTML display
        data = response.content if response.content else "No response from AI."
        return re.sub(r'\n', '<br>', data)
        
    except Exception as e:
        return f"Error during grading: AI model invocation failed. Details: {e}"
   

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/process', methods=['POST'])
def process_documents():
    global vectorstore, conversation_chain
    pdf_docs = request.files.getlist('pdf_docs')
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore)
    return redirect('/chat')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    global vectorstore, conversation_chain, chat_history
    msgs = []
    
    if request.method == 'POST':
        user_question = request.form['user_question']
        
        response = conversation_chain({'question': user_question})
        chat_history = response['chat_history']
        
    return render_template('chat.html', chat_history=chat_history)

@app.route('/pdf_chat', methods=['GET', 'POST'])
def pdf_chat():
    return render_template('pdf_chat.html')

@app.route('/essay_grading', methods=['GET', 'POST'])
def essay_grading():
    result = None
    text = ""
    global rubric_text

    if request.method == 'POST':
        #Handle rubric input safely
        rubric_text = request.form.get('essay_rubric', rubric_text)
        if not rubric_text.strip():
            rubric_text = """
            Default Rubric:
            - Clarity of ideas (30%)
            - Organization & structure (30%)
            - Grammar, spelling, and vocabulary (20%)
            - Creativity & originality (20%)
            """

        #Handle file input safely
        file = request.files.get('file')
        if file and file.filename.strip():
            text = extract_text_from_pdf(file)

        #If no file, get essay text from textarea
        else:
            text = request.form.get('essay_text', '').strip()

        #Check if text is empty
        if not text:
            result = "Please upload a PDF or enter essay text."
        else:
            result = _grade_essay(text, rubric_text)

    # Render template with current result and text (if any)
    return render_template('essay_grading.html', result=result, input_text=text)

    
@app.route('/essay_rubric', methods=['GET', 'POST'])
def essay_rubric():
    return render_template('essay_rubric.html')

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    return text

if __name__ == '__main__':
    app.run(debug=True)