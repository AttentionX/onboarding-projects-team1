import os
import re
from pathlib import Path
from datetime import datetime
import openai
from data import *
from process import *
from data import papers
import argparse

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv('OPENAI_API_KEY')

engine = "text-davinci-003"
max_tokens = 1000
temperature = 0.1


#load paper and preprocessing
paper = papers.transformer

parser = argparse.ArgumentParser()
parser.add_argument('--max_tokens', type=int, default=2000)
parser.add_argument('--temperature', type=float, default=0.7)
args = parser.parse_args()

abstract = re.findall(r'Abstract(.*?)[\n\d]*Introduction', paper, re.DOTALL)[0].strip()
abstract = " ".join(abstract.split('\n'))

paper_body = re.findall(r'Introduction(.*?)[\n\d]*References', paper, re.DOTALL)[0].strip()
paper_body = " ".join(paper_body.split('\n'))

print("paper length : ", len(paper_body))

split_length = 2000
total_length = len(paper_body)
paper_pieces = [paper_body[i:i+split_length] for i in range(0, total_length, split_length)]

num_pieces = len(paper_pieces)


#logging prompt and answers
log_path = Path(__file__).resolve().parent / "logs" / str(datetime.now())
log_path.mkdir(exist_ok=True)


#abstract prompt
abstract_prompt = f"""
This is the paper understanding task. You should read the abstract of the paper below and can memrize important information less than 10 sentences.
You can make less than 5 question that seems to be important to understand the paper but cannot be solved with provided context. 
This is the abstract of the paper:
{abstract}

Memorize important information and make questions and solve questions. Making less sentences and less questions as possible. Numbering and your important information sentences and questions.
Make a title 'Memory' and 'Questions' above your information sentences and questions.
Information about Human name is not important.
"""
with open(log_path / "prompt.txt", 'a') as fh:
    fh.write("abstract prompt : \n" + abstract_prompt + "\n")

response = openai.Completion.create(
    engine=engine, 
    prompt=abstract_prompt,
    temperature=temperature,
    max_tokens=max_tokens)

print(response.choices[0].text)

response_raw = response.choices[0].text
with open(log_path / "response.txt", 'a') as fh:
    fh.write(response_raw)

areas = response_raw.split("Questions:")
memory_area = areas[0]
question_area = "Questions : \n" + areas[1]

for i in range(num_pieces):
    body_prompt = f"""
    This is the paper understaning task. You will be provided small piece of paper, important information from the paper you made before, unsolved questions you made before.
    You should read the paper part and update 'Memory' area and 'Questions' area. 
    You can erase 1 sentences in 'Memory' area if it is not important.
    You also can add less than 2 new important information sentences, but max number of sentences in "Memory" area is 10.
    You can erase questions in "Question" area if they can be solved with provided information and add the answer information in your memory.
    You can make less than 2 question that cannot be solved with provided context, but max number of questions in "Questions" Area is 5.
    The paper is presented in {num_pieces} pieces. 

    This is the paper part {i+1} over {num_pieces} pieces.
    {paper_pieces[i]}

    This is the memory you made before.
    {memory_area}

    This is the questions you made before.
    {question_area}

    Update "Memory" and "Questions" area. Numbering and your important information sentences and questions. Memory should have less than 10 sentences. Questions should have less than 5 sentences.
    Make a title 'Memory' and 'Questions' above your information sentences and questions.
    Information about Human name is not important.
    """
    with open(log_path / "prompt.txt", 'a') as fh:
        fh.write(f"Body prompt {i}: \n" + body_prompt + "\n")

    response = openai.Completion.create(
    engine=engine, 
    prompt=body_prompt,
    temperature=temperature,
    max_tokens=max_tokens)

    response_raw = response.choices[0].text

    print(response_raw)
    with open(log_path / "response.txt", 'a') as fh:
        fh.write(response_raw)

    areas = response_raw.split("Questions:")
    memory_area = areas[0]
    question_area = "Questions : \n" + areas[1]



