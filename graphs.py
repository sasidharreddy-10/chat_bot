import os
import re
import time
import uuid
import PyPDF2
from typing import Any
import pandas as pd
from io import StringIO
from openai import OpenAI
from openai import AzureOpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter



class Generate_Graph:
    def __init__(self, index_name, PINECONE_API_KEY, gpt_engine_name, embedding_model_name, api_key, azure_endpoint, api_version, openai_type):
        self.index_name = index_name
        self.docsearch=None
        self.answer=''
        self.dataframes={}
        self.gpt_engine_name=gpt_engine_name
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

    def extract_text(self, file_path, file_name):
        try:
            excel_file=pd.ExcelFile(file_path)
            sheet_names=excel_file.sheet_names
            docs=[]
            for i in sheet_names:
                df = excel_file.parse(sheet_name=i)
                print("___________________________________________________________________________")
                print(type(df))
                print(df.head(5))
                print(file_name)
                print(i)
                file_path=i+".xlsx"
                df.to_excel(file_path, index=False)
                # df.to_excel()
                self.dataframes[file_name]=df
                mf=df.to_markdown(index=False)
                doc =  Document(page_content=mf, metadata={"filename": file_path, "page_no":"excel"})
                docs.append(doc)
            return docs
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            raise Exception("Error in Excel reading")
        
    def insert_docs(self, file_path, file_name):
        try:
            docs=self.extract_text(file_path, file_name)
            self.docsearch.add_documents(docs)
            return 'Successfully indexed the document'
        except Exception as e:
            print(f"Error splitting PDF text: {e}")

    def _create_graph_prompt(self, id, columns):
        python_prompt=f'''You are a senior python developer. Your task is to generate precise Plotly Python code to create visual graphs based on the provided dataframe sample data, dataframe name, dataframe columns, and user query.
Carefully analyse the given dataframe name, dataframe sample data, user query to write accurate python code.

Important points to note before generating code:
- The dataframe is already loaded in memory, so just use the given dataframe name in code. Never generate sample data just use the dataframe name.
- The graph should have accurate x_axis, y_axis labels with title. If the graph is stacked then show it in different colors with legend labels. The labels of the graph is very important. Title of the graph should be there at top or bottom of the graph.
- The minimum size of graph should be of height 700 and width 900.
- If the user asking about yearwise or quarterwise or monthwise data then show respective data only in the graph without extra information.
- In the generated python code, save the graph in html format to this path - 'temp/graph_{id}.html'

Follow the below steps to generate accurate python code for user query.

Step 1: Exhaustively analyse the given dataframe name, dataframe sample data, user query to write accurate python code.

Step 2: Based on the analysis in Step 1, select the type of graph that is most suitable for the given dataframe sample data according to the user query. Additionally, determine which columns are most relevant and should be considered for generating the graph as per the user query.
Only use the below available dataframe columns in code.
dataframe columns:
{columns}

Step 3: Select the most appropriate x-axis and y-axis from the dataframe based on the user query. These axes should clearly address the query and contain relevant values with appropriate labels.

Step 4: Based on the choices made in Steps 2 and 3, including the graph type, x-axis, and y-axis, think through the process step-by-step to generate accurate Python code that will create the desired graph using the given dataframe sample data, dataframe name, and user query.

Follow all the above steps and important points to generate precise python code.
'''
        return python_prompt
    
    def excecute_python_code(self,query, filters):
        try:
            id=uuid.uuid1().hex
            print(filters[0])
            df=pd.read_excel(filters[0])
            print(df.head())
            df.columns = df.columns.str.strip()
            print(type(df))
            print(df.dtypes)
            col=df.columns.to_list()
            python_prompt=self._create_graph_prompt(id, col)
            messages=[{'role':'system', 'content':python_prompt}]
            messages.append({'role': 'user', 'content': f'''Your task is to generate Python code using the Plotly library to create a graph based on the provided dataframe sample data, dataframe name, and user query. Use only the available dataframe columns in creating the graph.
    The dataframe is already loaded in memory, so just use the given dataframe name in code. Never read the dataframe, just use the dataframe name in code as it is already loaded in memory.

    Dataframe Name:
    df
                    
    Dataframe sample data:
    {df.head}
                            
    user query:
    {query}

    Do not generate any explanation for code just generate python code only.

    Python Code:
    '''})
            ans = self.openai_client.chat.completions.create(
                                model=self.gpt_engine_name,
                                messages=messages,
                            )
            res=ans.choices[0].message.content
            try:
                pattern = r"```python(.*?)```"
                matches = re.findall(pattern, res, re.DOTALL)
                if matches:
                    python_code = matches[0].strip()
                else:
                    raise
            except Exception as e:
                print("code parsing error occured")
                pattern = r'```(.*?)```'
                matches = re.findall(pattern, res, re.DOTALL)
                if matches:
                    python_code = matches[0].strip()
                else:
                    python_code=res
            try:
                print('##################')
                exec(python_code)
                print("@@@@@@@@@@@@ code excecution completed")
            except Exception as e:
                error=e
                print('code excecution error', e)
                retry_limit=2
                count=0
                messages=[
                        {'role': 'system', 'content': f'''You are a senior python developer. Your task is to correct the errors in the python code based on error message, dataframe name, dataframe sample data and user query. The python code is to generate visual graphs using plotly. Based on the given pyhton code and error message generate correct python code.
    The dataframe is already loaded in memory, so just use the given dataframe name in code. Never generate sample data just use the dataframe name.'''}
                        ]
                while count<retry_limit:
                    print("@@@@@@@@@@@", count)
                    messages.append({'role': 'user', 'content': f'''Your task is to correct the given Python code using the Plotly library to create a graph based on the provided error message, dataframe sample data, dataframe name, python code, and user query. Use only the available dataframe columns in creating the graph.
    Only use the dataframe name to create the graph. Only use the below dataframe name, dataframe sample data and user query in correcting code.

    Dataframe Name:
    df
                    
    Dataframe sample data:
    {df.head}

    python code:
    {python_code}

    error message:
    {error}
                            
    user query:
    {query}

    Do not generate any explanation for code just correct the above wrongly generated python code.

    Correct Python Code:
    '''})
                    response = self.openai_client.chat.completions.create(
                                model=self.gpt_engine_name,
                                messages=messages
                            )
                    res = response.choices[0].message.content
                    try:
                        pattern = r"```python(.*?)```"
                        matches = re.findall(pattern, res, re.DOTALL)
                        if matches:
                            python_code = matches[0].strip()
                        else:
                            raise
                    except Exception as e:
                        print("code parsing error occured")
                        pattern = r'```(.*?)```'
                        matches = re.findall(pattern, res, re.DOTALL)
                        if matches:
                            python_code = matches[0].strip()
                        else:
                            python_code=res
                    try:
                        exec(python_code)
                        print("code excecuted breaking ....................")
                        break
                    except Exception as e:
                        error=e
                    count+=1
                    messages.append({"role": "assistant", "content": python_code})
            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            print(python_code)
            return id
        except Exception as e:
            print("Error in excecute_python_code", e)