from langchain_openai import ChatOpenAI
from dotenv import load_dotenv  
from langchain_core.prompts import PromptTemplate
from langchain_core.tracers import ConsoleCallbackHandler
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
import os


# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")    
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")    

prompt = PromptTemplate(
    input_variables=["topic"],
    template="You are an expert AI Engineer. generate 5 interesting facts about : {topic}"
)

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o")

# create  parser

output_parser = StrOutputParser()

# Create the chain

chain = prompt  | llm | output_parser


result = chain.invoke({'topic': 'Langchain'})

print(result)

# Output the graph metadata
print("Graph Metadata:")
print(chain.get_graph().print_ascii())

exit(0)

# Output the trace run ID
#print(f"Trace Run ID: {chain.get_trace().get_run_id()}")

# output the call sequence 
print("Call Sequence:")
for call in chain.get_trace().get_calls():
    print(f"Call ID: {call.call_id}, Input: {call.input}, Output: {call.output}")
# Output the trace metadata
print("Trace Metadata:")
print(chain.get_trace().get_metadata())

chain.get_graph().get_nodes()

# Output the graph nodes
print("Graph Nodes:")
for node in chain.get_graph().get_nodes():
    print(f"Node ID: {node.node_id}, Type: {node.node_type}, Input Variables: {node.input_variables}, Output Variables: {node.output_variables}")

# Output the graph edges
print("Graph Edges:")
for edge in chain.get_graph().get_edges():
    print(f"Edge from {edge.source_node_id} to {edge.target_node_id}, Type: {edge.edge_type}")

# Output the graph metadata
print("Graph Metadata:")
print(chain.get_graph().print_ascii())


# Output the chain summary
print("Chain Summary:")
print(chain.get_summary())  