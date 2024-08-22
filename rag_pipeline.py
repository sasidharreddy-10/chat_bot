# import
import os
import time
import PyPDF2
from typing import Any
from openai import OpenAI
from openai import AzureOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    def extract_text(self, file_path, file_name):
        try:
            reader = PyPDF2.PdfReader(file_path)
            docs=[]
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text() if page.extract_text() else ""
                doc =  Document(page_content=text, metadata={"filename": file_name, "page_no":int(page_num)+1})
                docs.append(doc)
            return docs
        except Exception as e:
            print(f"Error reading PDF file: {e}")
            raise Exception("Error in PDF reading")
        
class RAG:
    def __init__(self, index_name, text_processor : TextProcessor, PINECONE_API_KEY, gpt_engine_name, embedding_model_name, api_key, azure_endpoint, api_version, openai_type) -> None:
        self.index_name = index_name
        self.docsearch=None
        self.answer=''
        self.gpt_engine_name=gpt_engine_name
        self.doc_processing=text_processor
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        if openai_type=='azure_openai':
            self.openai_client = AzureOpenAI(
                    azure_endpoint = azure_endpoint,
                    api_key=api_key,
                    api_version= api_version
                )
            self.embeddings = AzureOpenAIEmbeddings(
                model=embedding_model_name, 
                api_key=api_key, 
                azure_endpoint=azure_endpoint, 
                openai_api_version=api_version)
        else:
            self.openai_client = OpenAI(api_key=api_key)
            self.embeddings = OpenAIEmbeddings(model=embedding_model_name, api_key=api_key)
        self.__call__()
        
    def __call__(self) -> None:
        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)
        self.docsearch = PineconeVectorStore(index_name=self.index_name, embedding=self.embeddings)

    def insert_doc(self, file_path, file_name):
        try:
            docs=self.doc_processing.extract_text(file_path, file_name)
            self.docsearch.add_documents(docs)
            return 'Successfully indexed the document'
        except Exception as e:
            print(f"Error splitting PDF text: {e}")

    def _qna_helper(self, query, context):
        res = self.openai_client.chat.completions.create(model=self.gpt_engine_name,
                                                messages=[
                                                        {'role': 'system','content':f'''I am a Expert QnA Bot. Your task is to deliver structured responses to the user's query, strictly based on the information provided within the Context. The Context is enclosed by triple asterisk(*) and user query is enclosed by triple backtick(`).
Principle: Use only the information provided in the Context as the definitive source to answer the user's query. Ensure that responses are strictly based on the Context.
Output Format: Provide the answer to the user query based on the given context. At the end, include the document_name and page_no from which the answer was extracted. If the answer was extracted from multiple document_name and page_no (numerical value), list each on a new line under the heading 'References.'

Important points:
1. If the answer can not be sourced from Context then reply with "Sorry, this information is out of my uploaded knowledge base, Please ask queries from Uploaded Documents." in "answer" key.
2. Structure the answer using following formating : **Relevant heading** > sub-headings >  numerical pointers > bullet pointers. All the headings should be in bold letters.

Context:
***{context}***
 
++++
 
Search for Answers:
 
Thought: Thoroughly analyze the provided user query, identify the intent using key words and search for the answer specifically within the given Context.
Action: Analyze the user query thoroughly and search for accurate answer in each section from Context. Ensure the answer is direct answer and addresses most parts of the user query. The answer should be concise, to the point without any extra information. The answer should be well-formatted, featuring multiple paragraphs separated by line breaks. The key points in answer should be highlighted using numerical points (- for sub pointers). If direct answer not found in Context then reply with"Sorry, this information is out of my uploaded knowledge base, Please ask queries from Uploaded Documents." in answer key.
Action Input: Search for the answer exhaustively in above Context.
Observation: Ensure responses are sourced exclusively only from Context and cover all relevant points required to answer user query. Ensure that the format of answer follow this structure, **Relevant heading** > sub-headings >  numerical pointers > bullet pointers. Make sure all headings are in bold letters. Do not mention any references to document name, page number and document context from which it is extracted in the 'answer'. Include these references only in the "document_name:page_no" key. 
  
Important points to note before generating Response:
1. Ground Truth for 'answer': Only use information in Context as ground truth to answer user query. Strictly answer need to be extracted and sourced from only given above Context. The Context has multiple sections seperated by (-----------), with each section contains document_name, page_no, section text. Search for the answer exhaustively in Context from each section. If answer to the user query is not present in Context then my Response will be "Sorry, this information is out of my uploaded knowledge base, Please ask queries from Uploaded Documents." with no References.
2. Structure to maintain in 'answer': Structure the answer using following formating : **Relevant heading** > sub-headings >  numerical pointers > bullet pointers. Format the answer with multiple paragraphs seperated by line breaks with numerical pointers to highlight key points.
3. Response in References: In the References the document_name is the name of the document from which answer is extracted and page_no is page_no from which the answer is extracted. In References mention them in this format: document_name: page_no. If the answer was extracted from multiple document_name and page_no, list each on a new line under the heading 'References.'. Only include numerical number in page_no. 
4. The answer should be concise and directly address the user's query. No extra or additional information should be included beyond the answer to the user's query.

Strictly follow all the above instructions before generating structured Response.
'''},  
                                                        {'role': 'user',  'content':f'''Based on the given below User Query extract the detailed, concise answer from Context and format it in the structure mentioned in above instructions. Structure the answer using following formating : **Relevant heading** > sub-headings >  numerical pointers > bullet pointers. Structure the answer with multiple paragraphs separated by line breaks, highlight key points using numerical pointers.
The answer should be 'precise, concise and directly address' the user's query. No extra or additional information should be included beyond the answer to the user's query.
Give a 'concise answer', where every line of it is exculsively 'sourced from Context'. In the response, "References" should include the document_name, which is the name of the document from which the answer was extracted, and the page_no, indicating the page_no from which answer is extracted. The format should be "document_name:page_no". Only include numerical number in page_no(numerical value).     
If the answer is extracted from multiple sources, please list all the document names and page numbers from which the information is drawn. "Extract the complete answer and give all the references from which answer is extracted."
                                                         
Think step by step and give response.
                                                                                                                  
User Query: ```{query}```
Response:
'''}],
                                                        stream=True
                                                )
        
        return res
    
    def qna(self, query, filters):
        if filters['filename']['$in']:
            docs = self.docsearch.similarity_search(query, k=7, filter=filters)
        else:
            docs = self.docsearch.similarity_search(query, k=7)
        context=''
        for i in docs:
            doc_name=i.metadata['filename']
            page_no=int(i.metadata['page_no'])
            text=i.page_content
            context+=f'document_name: {doc_name} \n\n' + f'page_no : {page_no}\n\n' + f'Section Text: {text}\n------------------------------------------------------------------------------------------------------------ \n\n'
        answer=self._qna_helper(query, context)
        final_answer=''
        for chunk in answer:
            if len(chunk.choices)>0 and chunk.choices[0].delta.content is not None:
                text = chunk.choices[0].delta.content
                final_answer+=text
                yield text
                time.sleep(0.02)
        self.answer=final_answer