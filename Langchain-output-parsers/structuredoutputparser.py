from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

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

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 fact about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'topic':'black hole'})

print(result)