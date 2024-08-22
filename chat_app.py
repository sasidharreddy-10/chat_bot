import os
import time
import asyncio
from io import BytesIO
import streamlit as st
from intent import Intent
from summarys import Summarization
from graphs import Generate_Graph
from rephrase import Rephrase
from rag_pipeline import RAG, TextProcessor

#Initialize TextProcessor class
textprocessor=TextProcessor(chunk_size=1000, chunk_overlap=20)

# Set an environment variable
os.environ['PINECONE_API_KEY'] = '193bfd5b-1a4a-4c8b-bc06-7ec7c4cfc66a'

PINECONE_API_KEY="193bfd5b-1a4a-4c8b-bc06-7ec7c4cfc66a"


# List of recommended questions
recommended_questions = [
    "Tell about baroda home loans",
    "Benifits of Baroda Home Loan",
    "statement of confidentiality?"
]

answers=['''Bank of Baroda offers Baroda Home Loan, which is designed for aspiring homeowners who dream of buying their own residence. This housing loan can be used for various purposes such as buying a plot, purchasing a flat, building your own home, or extending your existing residence. Baroda Home Loan offers several exclusive features and benefits, including low interest rates, low processing charges, higher loan amounts, free credit card, and longer tenures. The loan amount is approved based on the location and income of the applicants. Furthermore, there are no hidden charges, no pre-payment penalty, and the interest rate is linked to Baroda Repo Linked Lending Rate (BRLLR) of the bank, which is reset monthly. To find out more about the Baroda Home Loan interest rate and other details, you can use the Bob Home Loan Calculator.''',
         '''The benefits of Baroda Home Loan include:

1. Low interest rates. \\n
2. Low processing charges.\\n
3. Higher loan amount.\\n
4. Free credit card.\\n
5. Longer tenures.''',
'''The Statement of Confidentiality is a contractual agreement between DEWA, its employees, and project stakeholders. It reinforces DEWA's commitment to protecting sensitive information and ensures that the Business Requirements Document (BRD) and associated project materials remain secure. The statement outlines stringent confidentiality measures that must be adhered to in order to build trust with stakeholders and emphasize DEWA's commitment to data security and privacy.''']

# Set the page title and layout
st.set_page_config(page_title="Chatbot Interface", layout="wide")

# Initialize session state variables
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''
if 'endpoint' not in st.session_state:
    st.session_state.endpoint = ''
if 'version' not in st.session_state:
    st.session_state.version = ''
if 'model_name' not in st.session_state:
    st.session_state.model_name = ''
if 'embedding_model_name' not in st.session_state:
    st.session_state.embedding_model_name = ''
if 'openai_type' not in st.session_state:
    st.session_state.openai_type = ''
if 'selected_files' not in st.session_state:
    st.session_state.selected_files = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []


# Function to display the modal
def display_modal():
    with st.expander("Please enter your Model Details:", expanded=False):
        # User selection for OpenAI or Azure OpenAI
        service = st.selectbox("Select Service", ["OpenAI", "Azure OpenAI"])
        with st.form(key='model_details_form'):
            if service == "OpenAI":
                st.session_state.openai_type = "openai"
                st.session_state.api_key = st.text_input("API Key", type="password")
                st.session_state.model_name = st.text_input("Model Name")
                st.session_state.embedding_model_name = st.text_input("Embedding Model Name")
                st.warning("Use text-embedding-002 model as VectorDB is optimized for it.")
            elif service == "Azure OpenAI":
                st.session_state.openai_type = "azure_openai"
                st.session_state.api_key = st.text_input("API Key", type="password")
                st.session_state.endpoint = st.text_input("Endpoint")
                st.session_state.version = st.text_input("Version")
                st.session_state.model_name = st.text_input("Model Name")
                st.session_state.embedding_model_name = st.text_input("Embedding Model Name")
                st.warning("Use text-embedding-002 model as VectorDB is optimized for it.")

            if st.form_submit_button("Submit"):
                # Collect and display the input values
                st.success("Details submitted successfully!")

# Display the modal when the app starts
display_modal()

# Function to show notification
def show_notification(message, message_type="success"):
    # Define CSS styles for the notification
    css = f"""
    <style>
    .notification {{
        position: fixed;
        top: 0;
        right: 0;
        margin: 20px;
        padding: 10px 20px;
        color: white; /* Text color */
        border-radius: 5px;
        z-index: 1000; /* Ensure it's on top */
        background-color: {"#4CAF50" if message_type == "success" else "#f44336"}; /* Green for success, red for error */
    }}
    </style>
    """
    notification_placeholder = st.empty()
    notification_placeholder.markdown(f'{css}<div class="notification">{message}</div>', unsafe_allow_html=True)
    time.sleep(5)
    notification_placeholder.empty()


if st.session_state.api_key:
    openai_type=st.session_state.openai_type
    gpt_engine_name=st.session_state.model_name
    embedding_model_name=st.session_state.embedding_model_name
    azure_endpoint = st.session_state.endpoint
    api_key=st.session_state.api_key
    api_version= st.session_state.version
    rephrase_obj=Rephrase(gpt_engine_name, embedding_model_name, api_key, azure_endpoint, api_version, openai_type)
    intent_obj=Intent(gpt_engine_name, embedding_model_name, api_key, azure_endpoint, api_version, openai_type)
    graph_obj=Generate_Graph("indexdb1", PINECONE_API_KEY, gpt_engine_name, embedding_model_name, api_key, azure_endpoint, api_version, openai_type)
    summary_obj=Summarization("indexdb1", PINECONE_API_KEY, gpt_engine_name, embedding_model_name, api_key, azure_endpoint, api_version, openai_type)
    rag_obj=RAG("indexdb1", textprocessor, PINECONE_API_KEY, gpt_engine_name, embedding_model_name, api_key, azure_endpoint, api_version, openai_type)


# Initialize chat history in session state if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "input_key" not in st.session_state:
    st.session_state.input_key = 0

uploaded_files = None
selected_file = None


# Handle file uploads only when user interacts with the sidebar
with st.sidebar:
    uploaded_files = st.file_uploader('Upload files', type=['pdf', 'xlsx'], accept_multiple_files=True, label_visibility="hidden")

    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            if uploaded_file not in st.session_state.uploaded_files:
                bytes_data = uploaded_file.read()
                file_like_object = BytesIO(bytes_data)
                file_name = uploaded_file.name
                st.session_state.uploaded_files.append(uploaded_file)
                
                try:
                    if st.session_state.api_key:
                        show_notification(f"{uploaded_file.name} preprocessing started, Please wait for a while!")
                        if file_name.endswith(('.pdf')):
                            rag_obj.insert_doc(file_like_object, file_name)
                        else:
                            graph_obj.insert_docs(file_like_object, file_name)
                        show_notification(f"{uploaded_file.name} inserted successfully!")
                    else:
                        show_notification("Please enter your OpenAI credentials before uploading documents!", message_type='error')
                except Exception as e:
                    show_notification("Something went wrong while preprocessing, please try again!", message_type='error')
    
    st.write("Select files:")
    files_to_keep = []
    for uploaded_file in st.session_state.uploaded_files:
        is_checked = st.checkbox(uploaded_file.name, value=uploaded_file in st.session_state.selected_files)
        if is_checked:
            files_to_keep.append(uploaded_file)
            if uploaded_file not in st.session_state.selected_files:
                st.session_state.selected_files.append(uploaded_file)
        else:
            if uploaded_file in st.session_state.selected_files:
                st.session_state.selected_files.remove(uploaded_file)

    # Update the session state with the currently selected files
    st.session_state.selected_files = files_to_keep


# Main content area
st.markdown("### What can I help you with today?")

# Display the message
st.write("You can upload documents in the sidebar")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize session state for the selected question
if 'selected_question' not in st.session_state:
    st.session_state.selected_question = "What is up?"

def handle_button_click(question):
    st.session_state.selected_question = question

# Function to handle button click
def handle_button_click(question):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.selected_question = question
    st.session_state.messages.append({"role": "user", "content": question})
    # Trigger bot query
    for i, j in enumerate(recommended_questions):
            if j==question:
                answer=answers[i]
    cache_answer(answer)
    st.warning("The answer is from the cache!")

def cache_answer(answer):
    answer_list=answer.split()
    def stream_answer():
        for i in answer_list:
            yield i + " "
            time.sleep(0.02)
    st.write_stream(stream_answer)
    st.session_state.messages.append({"role": "Bot", "content": answer})


def process_input(prompt, rephrase_prompt):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    selected_files=st.session_state.selected_files
    selected_names=[]
    for file in selected_files:
        selected_names.append(file.name)
    print("__________________________________________________________ selected files", selected_names)
    filters={
  "filename": { "$in": selected_names }
}
    print("__________________________________________________________ filtered files", filters)
    try:
        if st.session_state.api_key:
            answer=rag_obj.qna(rephrase_prompt, filters)
            with st.chat_message("BOT"):
                st.write_stream(answer)
            # Add user message to chat history
            final_answer=rag_obj.answer
            st.session_state.messages.append({"role": "Bot", "content": final_answer})
            rag_obj.answer=''
        else:
            st.error("Please enter your API Key and other details at the top.")
    except Exception as e:
        print("error", e)
        answer="Something went wrong, please try again(also check your openai credentials)."
        st.error(answer)

def graph_input(prompt):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    selected_files=st.session_state.selected_files
    selected_names=[]
    for file in selected_files:
        selected_names.append(file.name)
    print("__________________________________________________________ selected files", selected_names)
    filters={
  "filename": { "$in": selected_names }
}
    print("__________________________________________________________ filtered files", filters)
    try:
        if st.session_state.api_key:
            if len(filters['filename']['$in'])==1:
                id=graph_obj.excecute_python_code(prompt, selected_names)
                print('id', id)
                html_path=f'temp/graph_{id}.html'
                print("htmlpath", html_path)
                with st.chat_message("BOT"):
                    with open(html_path, 'r', encoding='utf-8') as file:
                        html_content = file.read()
                    st.components.v1.html(html_content, height=600)
                # Add user message to chat history
                st.session_state.messages.append({"role": "Bot", "content": "Your gaph has been saved in temp folder. You can download it anytime!"})
            else:
                st.error("Please select a single excel to generate the graph.")
        else:
            st.error("Please enter your API Key and other details at the top.")
    except Exception as e:
        print("error", e)
        answer="Something went wrong, please check your openai credentials"
        st.error(answer)

async def summary_input(prompt):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    selected_files=st.session_state.selected_files
    selected_names=[]
    for file in selected_files:
        selected_names.append(file.name)
    print("__________________________________________________________ selected files", selected_names)
    filters={
  "filename": { "$in": selected_names }
}
    print("__________________________________________________________ filtered files", filters)
    try:
        if st.session_state.api_key:
            if len(filters['filename']['$in'])==1:
                summary=await summary_obj.summarize_document(rephrase_prompt, filters)
                with st.chat_message("BOT"):
                    st.markdown(summary)
                # Add user message to chat history
                st.session_state.messages.append({"role": "Bot", "content": summary})
            else:
                st.error("Please select a single file to generate summary")
        else:
            st.error("Please enter your API Key and other details at the top.")
    except Exception as e:
        print("error", e)
        answer="Something went wrong, please try again(also check your openai credentials)"
        st.error(answer)

# Display recommended questions as buttons
if not st.session_state.messages:
    st.write("**Recommended Questions:**")
    cols = st.columns(len(recommended_questions))
    for i, question in enumerate(recommended_questions):
        if cols[i].button(question):
            handle_button_click(question)

# Prefill input bar with the selected question if available
if prompt := st.chat_input(st.session_state.selected_question):
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        history=""
        for i in st.session_state.messages[-2:]:
            if i['role']=='user':
                history+=f"User: {i['content']}\n"
            if i['role']=='Bot':
                history+=f"Bot: {i['content']}\n"
        if len(st.session_state.messages)>2:
            rephrase_prompt=rephrase_obj.followup_query(prompt, history)
        else:
            rephrase_prompt=prompt
        print(rephrase_prompt)
        intent=intent_obj.predict_intent(prompt)
        print("###################################", intent)
        if intent=="graph":
            graph_input(prompt)
        elif intent=="summary":
            asyncio.run(summary_input(prompt))
        else:
            process_input(prompt, rephrase_prompt)


# Dummy element to trigger auto-scrolling
auto_scroll = st.empty()

# Trigger auto-scroll by adding an empty message after all other messages
with auto_scroll:
    st.write("")  # This empty write forces the page to render and auto-scroll
