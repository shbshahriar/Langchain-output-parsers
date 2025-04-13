from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os

os.environ['HF_HOME'] = 'D:/HF/Tiny LLama'  # 🗃️ Path where model + tokenizer files will be cached']

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',  # ✅ Small, efficient open-source chat model
    task='text-generation',                         # 📌 Specifies the task (can also be 'text2text-generation')

    # Customize inference behavior
    pipeline_kwargs=dict(
        temperature=0.5,       # 🎯 Controls randomness: 0 = more focused, 1 = more creative
        max_new_tokens=500,    # ✂️ Limit the response length
        do_sample=True         # 🔄 Use sampling for generation
        # top_k=50,            # 🔢 (Optional) Consider top-k tokens only
        # top_p=0.95,          # 📊 (Optional) Use nucleus sampling
        # repetition_penalty=1.1  # 🔁 (Optional) Penalize repeating words/phrases
    )
)
model=ChatHuggingFace(llm=llm)

class Person(BaseModel):

    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

final_result = chain.invoke({'place':'sri lankan'})

print(final_result)