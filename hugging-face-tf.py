
#pretrained models are downloaded and cached in C:\Users\username\.cache\huggingface\hub for windows


from transformers import pipeline
classifier = pipeline('sentiment-analysis')
print(classifier('we love you'))