from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

report_prompt=PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)
report_chain=report_prompt | model| StrOutputParser()

report_result= report_chain.invoke({'topic':'Black hole'})

print("=== Full Report ===")
print(report_result)

summery_prompt=PromptTemplate(
    template='write a exactly  5 line summery on following text not more than 5 line: {text}',
    input_variables=['text']
)


summery_chain=summery_prompt | model| StrOutputParser()
summery_result= summery_chain.invoke({'text':report_result})

print("=== Summery ===")
print(summery_result)