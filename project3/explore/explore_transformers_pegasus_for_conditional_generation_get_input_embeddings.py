from transformers import  PegasusForConditionalGeneration

model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-cnn_dailymail')
# explore the input word embeddings of the model
embeddings = model.get_input_embeddings()
# check the shape of the embeddings
print(embeddings.weight.shape)  # (|V|, H)

