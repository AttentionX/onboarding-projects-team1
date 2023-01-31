from pathlib import Path
import wandb
from transformers import PegasusTokenizer
from gpt2_soft import PegasusSoftConfig, PegasusSoftForConditionalGeneration
import streamlit

# load the model locally
artifact = wandb.Api().artifact("eubinecto/onboarding-projects-team1/pegasus:prod", type="model")
artifact_path = artifact.download()
tokenizer = PegasusTokenizer.from_pretrained(Path(artifact_path) / "tokenizer")
model_config = PegasusSoftConfig.from_pretrained(Path(artifact_path) / "model")
model = PegasusSoftForConditionalGeneration.from_pretrained(Path(artifact_path) / "model", config=model_config)
text = "Hi, How are you doing?"
# tokenize the text and generate
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


