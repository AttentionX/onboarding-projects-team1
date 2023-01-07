from pathlib import Path
from vaswani_2017 import PAPER
import re
import argparse
import openai
import os
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--max_tokens', type=int, default=2000)
parser.add_argument('--temperature', type=float, default=0.7)
args = parser.parse_args()
abstract = re.findall(r'Abstract(.*?)[\n\d]*Introduction', PAPER, re.DOTALL)[0].strip()
abstract = " ".join(abstract.split('\n'))

prompt = f"""
Title:
Attention is all you need

Abstract:
{abstract}

Read the title and abstract above, and answer the following questions:
1. What did authors try to accomplish? Describe with rich examples.
2. What were the key elements of the approach? Describe the mathematics behind the key elements as well with rich examples.
3. In what ways the approach was limited by? Describe with rich examples.
4. How could you use it for computer-assisted language learning? Describe with rich examples.
5. What other references should you follow? 

Answers:
""".strip()


openai.api_key = os.getenv('OPENAI_API_KEY')
response = openai.Completion.create(engine="text-davinci-003",
                                    prompt=prompt,
                                    temperature=args.temperature,
                                    max_tokens=args.max_tokens)
# print out the summary
summary = response['choices'][0]['text']
print(summary)
# log the summary to refer to later on
log_path = Path(__file__).resolve().parent / "logs" / str(datetime.now())
log_path.mkdir(exist_ok=True)
with open(log_path / "summary.txt", 'w') as fh:
    fh.write(summary)



