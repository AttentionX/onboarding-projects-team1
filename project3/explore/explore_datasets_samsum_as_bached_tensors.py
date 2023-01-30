from datasets import load_dataset

samsum_dataset = load_dataset('samsum')
# have a look
print(samsum_dataset['train'][0]['dialogue'])
print(type(samsum_dataset['train']))
