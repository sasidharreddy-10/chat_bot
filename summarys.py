import os
import re
import time
import uuid
import asyncio
import nest_asyncio
import random
import PyPDF2
import tiktoken
from typing import Any
from openai import OpenAI, AsyncOpenAI
from openai import AzureOpenAI, AsyncAzureOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

nest_asyncio.apply()

class Summarization:
    def __init__(self, index_name, PINECONE_API_KEY, gpt_engine_name, embedding_model_name, api_key, azure_endpoint, api_version, openai_type):
        self.index_name = index_name
        self.docsearch=None
        self.answer=''
        self.tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo-16k-0613')
        self.gpt_engine_name=gpt_engine_name
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        if openai_type=='azure_openai':
            self.openai_client = AzureOpenAI(
                    azure_endpoint = azure_endpoint,
                    api_key=api_key,
                    api_version= api_version
                )
            self.async_openai_client = AsyncAzureOpenAI(
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
            self.async_openai_client = AsyncOpenAI(api_key=api_key)
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

    def text_splitter(self, text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=100
        )
        return splitter.split_text(text)

    async def chat_completion(self, messages: list, temperature: float, model_name: str, max_token: int) -> str:
        response = "OpenAI Not Responding"
        for delay_secs in (2**x for x in range(0, 2)):
            try:
                response = await self.async_openai_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_token
                )
                return response.choices[0].message.content
            except Exception as e:
                random_value = random.randint(0, 1000) / 1000.0
                sleep_dur = delay_secs + random_value
                time.sleep(sleep_dur)
        return response


    def _summary_prompt(self, context):
        system_prompt = f"""Your Task is to provide a very detailed and comprehensive summary of the given Document. Summarize the given document text delimited by triple backtick(`), from a first-person perspective.

Your summary should adhere to the following guidelines:
1. Act as a detailed-oriented professional summarizer, Consider the audience as someone with a basic understanding of document. Provide a detailed summary that emphasizes on key points of document.
2. Use the text in Document as the definitive reference and only use its text to create the summary.
4. Ensure the explanation is thorough and point to point.
5. Exclude any concluding statements at the end of the summary.

Follow the below summary structure:
1. Use paragraphs, numerical pointers and bullet points separated by line breaks.
2. In summary follow this formating : Relevant heading > sub-headings > bullet pointers > numerical pointers

Strictly use the provide context and provide the detailed summary using the above guidelines
"""
        messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content":f'''Please summarize the following document from a first-person perspective. Consider only below document text for summarization. If the document has no text then reply no content in the document.
Do not include any concluding statements at the end of the summary. Ensure that all main points from the content are included in the summary.
Document:
```{context}```
++++
summary:'''}
                    ]
        
        return messages
    
    async def summarize_chunk(self, chunk: str, chunk_length: int) -> str:
        messages = self._summary_prompt(chunk)
        chunk_summary=await self.chat_completion(messages, 0.1, self.gpt_engine_name, 1000)
        return chunk_summary
    
    async def summarize_document(self, query, filters) -> str:
        docs = self.docsearch.similarity_search(query, k=100, filter=filters)
        content=f"Document Name: {filters['filename']}\n\n"
        for i in docs:
            text=i.page_content
            content+=f'Section Text: {text}\n\n'
        print(content)
        doc_tokens = len(self.tokenizer.encode(content))
        summary_tokens=doc_tokens
        if doc_tokens >= 5000:
            while summary_tokens>doc_tokens*0.20:
                chunks = self.text_splitter(content)
                tasks = [self.summarize_chunk(chunk, len(chunks)) for chunk in chunks]
                summaries = await asyncio.gather(*tasks)
                content='.\n\n'.join(summaries)
                summary_tokens = len(self.tokenizer.encode(content))
        else:
            summaries=[]
            messages = self._summary_prompt(content)
            content=await self.chat_completion(messages, 0.1, self.gpt_engine_name, 1000)
 
        return content