import json
from pathlib import Path

from revChatGPT.ChatGPT import Chatbot

from vaswani_2017 import PAPER
import re
import argparse
import openai
import os
from datetime import datetime


abstract = re.findall(r'Abstract(.*?)[\n\d]*Introduction', PAPER, re.DOTALL)[0].strip()
abstract = " ".join(abstract.split('\n'))
intro = re.findall(r'Introduction(.*?)[\n\d]*Background', PAPER, re.DOTALL)[0].strip()
intro = " ".join(intro.split('\n'))
background = re.findall(r'Background(.*?)[\n\d]*Model Architecture', PAPER, re.DOTALL)[0].strip()
background = " ".join(background.split('\n'))

chatbot = Chatbot({
  "session_token": "<YOUR_TOKEN>"
})

starter = """
I want you to answer some questions I have with regards to a paper.  But first, since you may not know what the paper is about, I'll have you read the paper, section by section. As you read, try remembering all sections, and answer my questions at the the end.
Are you with me on this?
""".strip()

chatbot.ask(starter)
for s in (abstract, intro, background):
    order = f"""
    {s}
    ---
Good. Here is the next section of the paper. Read and remember this. I'll show you the next section of the paper shortly after you finish reading this.
Have you finished reading this? Just say yes or no.
    """
    chatbot.ask(order)
    check = """
    How many sections have you read so far? What were the key points of each section? Keep your answer short and concise.
    """
    chatbot.ask(check)


# then ... we ask questions
questions = [
    "What is the main idea of the paper?",
    "What is the main contribution of the paper?",
    "What is the main problem the paper is trying to solve?",
    "What is the main result of the paper?",
    "What is the main takeaway of the paper?",
]


for q in questions:
    response = chatbot.ask(q)

# log the summary to refer to later on
log_path = Path(__file__).resolve().parent / "logs" / str(datetime.now())
log_path.mkdir(exist_ok=True)
with open(log_path / "response.json", 'w') as fh:
    fh.write(json.dumps(response)) # noqa



