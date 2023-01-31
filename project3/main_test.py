from pathlib import Path
import wandb
from transformers import BartTokenizer
from bart_soft import BartSoftConfig, BartSoftForConditionalGeneration
import streamlit

# load the model locally
artifact = wandb.Api().artifact("eubinecto/onboarding-projects-team1/bart:latest", type="model")
artifact_path = artifact.download()
tokenizer = BartTokenizer.from_pretrained(Path(artifact_path) / "tokenizer")
model_config = BartSoftConfig.from_pretrained(Path(artifact_path) / "model")
model = BartSoftForConditionalGeneration.from_pretrained(Path(artifact_path) / "model", config=model_config)
text = "Hi, How are you doing?"
# tokenize the text and generate
inputs = tokenizer([text], return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


