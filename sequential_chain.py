from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.tracers import ConsoleCallbackHandler
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser



# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")      
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")    
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")


prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="You are an expert AI Engineer. generate short summary about : {topic}"
)

prompt2 = PromptTemplate(
    input_variables=["text"],
    template="You are an expert AI Engineer. generate 5 pointer summary from the following  text  : {text}"
)

#LLM initialization
llm = ChatOpenAI(model="gpt-4o")

# create  parser
output_parser = StrOutputParser()

# Create the chain

chain = prompt1 | llm | output_parser | prompt2  | llm | output_parser

result = chain.invoke({'topic': 'Unemplyemt in India'})

print(result)

chain.get_graph().print_ascii()