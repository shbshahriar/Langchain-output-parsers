from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
import json

# Set the path to where the Hugging Face model and tokenizer are stored
os.environ['HF_HOME'] = 'D:/HF/Tiny LLama'

# Initialize the Hugging Face pipeline with the TinyLlama model
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',  # Chat model from TinyLlama
    task='text-generation',                        # Specify the task (text-generation)
    pipeline_kwargs=dict(
        temperature=0.5,                           # Controls randomness
        max_new_tokens=500,                       # Limits response length
        do_sample=True                             # Use sampling for generation
    )
)

# Initialize the model with ChatHuggingFace
model = ChatHuggingFace(llm=llm)

# Initialize the JSON output parser
parser = JsonOutputParser()

# Create a prompt template for the model
template = PromptTemplate(
    template='Give me name, age, and city of a fictional character. Please return the response in the following format: {"name": "Name", "age": Age, "city": "City"}',
    input_variables=[],
)

# Format the prompt with the template
prompt = template.template

# Invoke the model with the formatted prompt
result = model.invoke(prompt)

# Clean the raw output (strip unnecessary text)
raw_output = result.content.strip()

# Clean the output to get only the JSON part (if there's additional text)
json_part = raw_output.split('Response: ')[-1].strip()

# Try parsing the JSON part
try:
    final_result = json.loads(json_part)
    print("Parsed JSON:", final_result)
except json.JSONDecodeError as e:
    print(f"Error while parsing JSON: {e}")
