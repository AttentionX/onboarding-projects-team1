import os
import openai
import re
from pathlib import Path
from datetime import datetime
import openai
from data import papers
import argparse
from haystack.document_stores import InMemoryDocumentStore
from haystack import Document
from haystack.nodes import PreProcessor, EmbeddingRetriever

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv('OPENAI_API_KEY')

engine = "text-davinci-003"
max_tokens = 1000
temperature = 0.1

#logging the conversation and prompt
log_path = Path(__file__).resolve().parent / "logs" / str(datetime.now())
log_path.mkdir(exist_ok=True)

#parse the paper's abstract and body
paper = papers.transformer
# paper = papers.retro

parser = argparse.ArgumentParser()
parser.add_argument('--max_tokens', type=int, default=2000)
parser.add_argument('--temperature', type=float, default=0.7)
args = parser.parse_args()

abstract = re.findall(r'Abstract(.*?)[\n\d]*Introduction', paper, re.DOTALL)[0].strip()
abstract = " ".join(abstract.split('\n'))

paper_body = re.findall(r'Introduction(.*?)[\n\d]*References', paper, re.DOTALL)[0].strip()
paper_body = "1  Introduction  "+" ".join(paper_body.split('\n'))


#create document store
document_store = InMemoryDocumentStore(
    index="document",
    embedding_field="emb",
    embedding_dim=768,
    similarity="dot_product",
)

#create preprocessor
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,
    split_by="word",
    split_length=100,
    split_overlap=30,
    split_respect_sentence_boundary=True,
    language="en"
)

#write document store
paper_body_dict = {
    "id": 1,
    "content": paper_body,
    "meta": {
        "name": "paper"
    }
}
doc = [Document.from_dict(paper_body_dict)]
docs = preprocessor.process(doc)

print(len(doc[0].content), len(docs))
document_store.write_documents(docs)

#create retriever
retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    model_format="sentence_transformers",
    progress_bar=False,
)
document_store.update_embeddings(retriever)


context = ''
response_start = "\nA : "
past_qa = []

index = 0

initial_prompt = f"""
    Q : You are a chatbot that can answer the questions about the paper. This is the abstract of the paper you should understand.
    {abstract}
    A : I understand the abstract. I'm ready to answer the questions.

    """

#run the chatbot
while True:
    print("Ask Question : ")
    user_input = input()
    if user_input == "exit":
        break

    sample_chunks = retriever.retrieve(query=user_input, top_k=5)
    with open(log_path / "retrieval.txt", 'a') as fh:
        fh.write("Question " + str(index) + " : " + user_input + "\n")
        fh.write("Answer : \n")
        for chunk in sample_chunks:
            fh.write(chunk.content + "\n")
        fh.write("\n\n")

    prompt = initial_prompt

    for qa in past_qa:
        prompt += "Q : " + qa["question"] + "\n"
        prompt += "A : " + qa["answer"] + "\n"

    prompt += "Q : " + user_input + "\n These informations from the paper below might be helpful.\n"
    for chunk in sample_chunks:
        prompt += chunk.content + "\n"
    
    prompt += "Answer like above "
    prompt += response_start
    # print(prompt)

    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    past_qa.append({"question": user_input, "answer": response.choices[0].text})
    if len(past_qa) >= 10:
        past_qa.pop(0)

    print("A : " + response.choices[0].text)
    index += 1

print("-----------------------------------------------")
print("Whole conversation")
print("-----------------------------------------------")
for qa in past_qa:
    print("Q : " + qa["question"])
    print("A : " + qa["answer"])
    with open(log_path / "conversation.txt", 'a') as fh:
        fh.write("Q : " + qa["question"] + "\n")
        fh.write("A : " + qa["answer"] + "\n")