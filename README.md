# Automatic Fact Cheking

# Methods
In this project, we built an automated fact-checking system designed for climate science claims. Our goal is to provide a more accurate and efficient method to address the prevalence of misinformation in this domain. The system is divided into two tasks: (1) to extract the most relevant evidence for each claim, using an unsupervised TF-IDF method and a fine-tuned neural network with BERT models; and (2) to classify the relationship between claims and evidence, using Roberta embeddings as inputs into a classifier.

# Results
To evaluate the system, we used F1-score for (1) and accuracy for (2) and determined the harmonic mean of these scores to rank in the Codalab competition. Our automated fact-checking system achieved a harmonic mean of 0.12, demonstrating its effectiveness in retrieving relevant evidence and classifying the relationship between claims and evidence.
