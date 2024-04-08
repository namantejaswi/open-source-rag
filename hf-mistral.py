# Import necessary classes from the Transformers library
from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import pipeline

import logging, sys
# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

 

# Load the tokenizer and model
#tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
#model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# Define the directory to save the model
save_directory = "models"

# Save the tokenizer and model to the specified directory
#Run once
#model.save_pretrained(save_directory)
#tokenizer.save_pretrained(save_directory)

# Load the tokenizer and model from the saved directory
tokenizer = AutoTokenizer.from_pretrained(save_directory, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(save_directory, local_files_only=True)    



pipe = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1")

# Generate text using the pipeline
result = pipe("attention is all you need", max_length=30, num_return_sequences=5, truncation=True)



