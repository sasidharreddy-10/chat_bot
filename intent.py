import os
import re
import time
import uuid
import PyPDF2
from typing import Any
from openai import OpenAI
from openai import AzureOpenAI
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings


class Intent:
    def __init__(self,  gpt_engine_name, embedding_model_name, api_key, azure_endpoint, api_version, openai_type):
        self.gpt_engine_name=gpt_engine_name
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

    def _intent_prompt(self):
        system_prompt='''You are an intent classification agent. Your task is to identify the intent of the provided user query. Carefully analyze the query and classify it into one of the following intents.

Intents Definitions:
graph: If the user query explicitly requests the generation of any type of graph from data, classify the intent of the query as "graph".
summary: If a user query 'explicitly ask for a summary' of a topic or document, classify the intent of that query as "summary". Only classify it as summary intent if the summary word is explicitly mentioned in the user query.
normal: If the user query is a general question using words like "where," "what," "tell," etc., and does not fit into the "summary" or "graph" intents, classify the intent as "other."

Carefully analyze the given user query along with the above intent definitions provided, and classify the query as either "graph", "summary" or "normal" based on its content.

Output only one of the following intents: graph, summary or normal. No additional explanation or information is allowed.
'''
        return system_prompt
    
    def predict_intent(self, query):
        system_prompt=self._intent_prompt()
        res = self.openai_client.chat.completions.create(model=self.gpt_engine_name,
                                                        messages=[
                                                                {'role': 'system','content':system_prompt},  
                                                                {'role': 'user',  'content':f'''Based on the user query provided, classify its intent. Analyze the intent definitions carefully before determining whether the intent is "summary" "graph" or "normal"
Only classify it as summary intent if the summary or summarize words are 'explicitly mentioned' in the user query. Strictly check for words like "summarize," "summarise," and similar terms, and classify the intent as "summary" only if these words are present in the user query.
Output only one of the following intents: graph, summary, or normal.

user query:
{query}

Think step by step and identify the intent of above user query.

Intent:
'''}],
                                                        temperature=0.1,
                                                        max_tokens=1
                                                        )
        
        intent=res.choices[0].message.content
        return intent