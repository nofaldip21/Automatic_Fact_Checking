#!/usr/bin/env python
# coding: utf-8

# Training The Bert Model

# In[1]:



import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import math
import pandas as pd
import pickle
import time
from transformers import BertModel
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

#list all stopword and punctuation
stopWords = set(list(stopwords.words('english')) + list(string.punctuation))
lemmatizer = WordNetLemmatizer()

# function to lemmatize word
def processLemma(text) :
    wordLemma = lemmatizer.lemmatize(text, 'v')
    if wordLemma == text:
        wordLemma = lemmatizer.lemmatize(text, 'n')
    return wordLemma  

# funstion to pre processing text
def textPreprocess(text):
    wordTokenize = word_tokenize(text)
    wordTokenize = [word.lower() for word in wordTokenize if (word.lower() not in stopWords) & (word.lower().isalpha())]
    lemmatizedTokens = [processLemma(word) for word in wordTokenize if len(processLemma(word)) > 1 ]
    return " ".join(lemmatizedTokens)


# In[2]:


# load section
# please change the folder path depend on your path location
with open("/Folder dataset/train-claims.json", 'r') as f:
    trainData = json.load(f)
with open('/TF-IDF embedding and vectorize/vectorize_tfIdf_without_new.pkl', 'rb') as handle:
    vectorizer = pickle.load(handle)
with open('/TF-IDF embedding and vectorize/evidence_list.pkl', 'rb') as handle:
    evidenceList = pickle.load(handle)
# cannot load this file to github because too large
with open('/TF-IDF embedding and vectorize/embedding_tfidf_without_new.pkl', 'rb') as handle:
    tfIdfMatrix = pickle.load(handle)
# cannot load this file to github because too large
with open('/Folder dataset/evidence.json','r') as f :
    dictEvidence = json.load(f)


# Generate Class Dataset

# In[3]:


from torch.utils.data import Dataset

class TrainedDataset(Dataset):

    def __init__(self, filename,maxlen):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.maxlen = maxlen  
        self.df = pd.read_csv(filename,sep="\t")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the data frame
        target_sentence = textPreprocess(self.df.loc[index, 'target_sentence'])
        evidence_sentence = textPreprocess(self.df.loc[index, 'evidence_sentence'])
        label = self.df.loc[index, 'label']

        #Preprocessing the text to be suitable for BERT
        halfLen = math.ceil(self.maxlen/2) - 1
        tokens_target = self.tokenizer.tokenize(target_sentence) #Tokenize the sentence
        seg_ids_target = [0 for _ in range(len(tokens_target))]
        tokens_evidence = self.tokenizer.tokenize(evidence_sentence)
        seg_ids_evidence = [1 for _ in range(len(tokens_evidence))] 
        tokens = ['[CLS]'] + tokens_target[:halfLen] + ['[SEP]'] + tokens_evidence[:halfLen] + ['[SEP]']#Insering the CLS and SEP token in the beginning and end of the sentence
        seg_ids = [0] + seg_ids_target[:halfLen] + [0] + seg_ids_evidence[:halfLen] + [1]
        if len(tokens) < self.maxlen:
            seg_ids = seg_ids + [1 for _ in range(self.maxlen - len(tokens))]
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))] #Padding sentences
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]'] #Prunning the list to be of specified max length
            seg_ids = seg_ids[:self.maxlen]

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) #Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids_tensor = torch.tensor(tokens_ids) #Converting the list to a pytorch tensor
        seg_ids_tensor = torch.tensor(seg_ids)

        #Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask,seg_ids_tensor, label


# Generate Sentiment Classification Class

# In[4]:


import torch
import torch.nn as nn
from transformers import BertModel

class SentimentClassifier(nn.Module):

    def __init__(self):
        super(SentimentClassifier, self).__init__()
        #Instantiating BERT model object 
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        
        #Classification layer
        #input dimension is 768 because [CLS] embedding has a dimension of 768
        #output dimension is 1 because we're working with a binary classification problem
        self.cls_layer = nn.Linear(768, 1)

    def forward(self, seq, attn_masks,seg_ids):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        outputs = self.bert_layer(seq, attention_mask = attn_masks,token_type_ids = seg_ids, return_dict=True)
        cont_reps = outputs.last_hidden_state

        #Obtaining the representation of [CLS] head (the first token)
        cls_rep = cont_reps[:, 0]

        #Feeding cls_rep to the classifier layer
        logits = self.cls_layer(cls_rep)

        return logits


# In[5]:


net = SentimentClassifier()
# saved_model = torch.load(folderPath + 'Saved Model/sstcls_1_512_balance_modif.dat')
# net.load_state_dict(saved_model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    net = nn.DataParallel(net)


# In[6]:


def get_accuracy_from_logits(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.2).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc

# Calculate real F1 not the mean
def get_accuracy_F1_element(logits, labels) :
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.2).long()
    TP = torch.sum((soft_probs.squeeze() == 1) & (labels == 1))
    FP = torch.sum((soft_probs.squeeze() == 1) & (labels == 0))
    FN = torch.sum((soft_probs.squeeze() == 0) & (labels == 1))
    return TP,FP,FN

# Mean of F1
def calculateTheF1(logits, labels):
    probs = torch.sigmoid(logits.unsqueeze(-1))
    soft_probs = (probs > 0.2).long()
    TP = torch.sum((soft_probs.squeeze() == 1) & (labels == 1))
    FP = torch.sum((soft_probs.squeeze() == 1) & (labels == 0))
    FN = torch.sum((soft_probs.squeeze() == 0) & (labels == 1))
    if (TP + FN) == 0 :
      recall = 0
    else :
      recall = TP / (TP + FN)
    if (TP + FP) == 0 :
      precision = 0
    else :
      precision = TP / (TP + FP)
    if (precision + recall) == 0 :
      F1 = 0
    else :
      F1 = 2 * precision * recall / (precision + recall)
    return F1

def evaluate(net, criterion, dataloader,device):
    net.eval()

    mean_acc, mean_loss = 0, 0
    TP,FP,FN = 0,0,0
    F1 = 0
    count = 0

    with torch.no_grad():
        for seq, attn_masks,seg_ids, labels in dataloader:
            seq, attn_masks,seg_ids, labels = seq.to(device), attn_masks.to(device),seg_ids.to(device), labels.to(device)
            logits = net(seq, attn_masks,seg_ids)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            mean_acc += get_accuracy_from_logits(logits, labels)
            F1 += calculateTheF1(logits, labels)
            TP_Temp,FN_temp,FP_temp = get_accuracy_F1_element(logits, labels)
            TP += TP_Temp
            FN += FN_temp
            FP += FP_temp
            count += 1
    F1_result = TP / (TP+0.5*(FP+FN))
    #print(count)
    return mean_acc / count, mean_loss / count,F1 / count,F1_result


# In[7]:


import time

def train(net, criterion, opti, train_loader, dev_loader, max_eps, device):

    best_acc = 0
    st = time.time()
    for ep in range(max_eps):
        
        net.train()
        for it, (seq, attn_masks,seg_ids, labels) in enumerate(train_loader):
            #Clear gradients
            opti.zero_grad()  
            #Converting these to cuda tensors
            seq, attn_masks, seg_ids ,labels = seq.to(device), attn_masks.to(device),seg_ids.to(device), labels.to(device)

            #Obtaining the logits from the model
            logits = net(seq, attn_masks,seg_ids)

            #Computing loss
            loss = criterion(logits.squeeze(-1), labels.float())


            #Backpropagating the gradients
            loss.backward()

            #Optimization step
            opti.step()
              
            if it % 100 == 0:
                
                acc = get_accuracy_from_logits(logits, labels)
                F1_result = calculateTheF1(logits, labels)
                print("Iteration {} of epoch {} complete. Loss: {}; Accuracy: {}; F1-result: {}; ;Time taken (s): {}".format(it, ep, loss.item(), acc,F1_result, (time.time()-st)))
                st = time.time()

        
        dev_acc, dev_loss,F1_result,_ = evaluate(net, criterion, dev_loader, device)
        print("Epoch {} complete! Development Accuracy: {}; Development Loss: {}; Development F1: {}".format(ep, dev_acc, dev_loss,F1_result))
        if F1_result > best_acc:
            print("Best development accuracy improved from {} to {}, saving model...".format(best_acc, F1_result))
            best_acc = F1_result
            torch.save(net.state_dict(),'sstcls_{}_512_for_100_wiki_model.dat'.format(ep))


# In[9]:


from torch.utils.data import DataLoader
import numpy as np

#Creating instances of training and development set
#maxlen sets the maximum length a sentence can have
#any sentence longer than this length is truncated to the maxlen size
train_set = TrainedDataset(filename = 'saved dataset/train-claims-dataset-cosine-100.csv', maxlen = 60)
dev_set = TrainedDataset(filename = 'saved dataset/dev-dataset-cosine-100.csv', maxlen = 60)


#Creating intsances of training and development dataloaders
train_loader = DataLoader(train_set, batch_size = 100, num_workers = 2)
dev_loader = DataLoader(dev_set, batch_size = 100, num_workers = 2)

print("Done preprocessing training and development data.")


# In[10]:


import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

criterion = nn.BCEWithLogitsLoss()
opti = optim.Adam(net.parameters(), lr = 2e-5)


# If train data is needed

# In[ ]:


num_epoch = 2

#fine-tune the model
train(net, criterion, opti, train_loader, dev_loader, num_epoch,device)


# Generate Test dataset

# In[12]:


def predict(net,dataloader,top10Value,device):
    net.eval()
    logitsList = []
    with torch.no_grad():
        for seq, attn_masks,seg_ids in tested_loader:
            seq, attn_masks,seg_ids = seq.to(device), attn_masks.to(device),seg_ids.to(device)
            logits = net(seq, attn_masks,seg_ids)
            probs = torch.sigmoid(logits.unsqueeze(-1))
            logitsList += probs.squeeze().tolist()
    combinedList = sorted(zip(logitsList, evidence_test), reverse=True)
    top5Elements = [e for s, e in combinedList[:5]]
    if len(top5Elements) == 0 :
      top5Elements = [e for s, e in combinedList[:5]]
    topSentence = [dictEvidence[e] for e in top5Elements]
    return top5Elements , topSentence


# Build Test Class

# In[13]:


from torch.utils.data import Dataset

class TestedDataset(Dataset):

    def __init__(self, df, maxlen):

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.maxlen = maxlen  
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the data frame
        target_sentence = self.df.loc[index, 'target_sentence']
        evidence_sentence = self.df.loc[index, 'evidence_sentence']

        #Preprocessing the text to be suitable for BERT
        halfLen = math.ceil(self.maxlen/2) - 1
        tokens_target = self.tokenizer.tokenize(target_sentence) #Tokenize the sentence
        seg_ids_target = [0 for _ in range(len(tokens_target))]
        tokens_evidence = self.tokenizer.tokenize(evidence_sentence)
        seg_ids_evidence = [1 for _ in range(len(tokens_evidence))] 
        tokens = ['[CLS]'] + tokens_target[:halfLen] + ['[SEP]'] + tokens_evidence[:halfLen] + ['[SEP]']#Insering the CLS and SEP token in the beginning and end of the sentence
        seg_ids = [0] + seg_ids_target[:halfLen] + [0] + seg_ids_evidence[:halfLen] + [1]
        if len(tokens) < self.maxlen:
            seg_ids = seg_ids + [1 for _ in range(self.maxlen - len(tokens))]
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))] #Padding sentences
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]'] #Prunning the list to be of specified max length
            seg_ids = seg_ids[:self.maxlen]

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) #Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids_tensor = torch.tensor(tokens_ids) #Converting the list to a pytorch tensor
        seg_ids_tensor = torch.tensor(seg_ids)

        #Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask,seg_ids_tensor


# Generate Top 100 for testing dataset

# In[14]:


def top100(claim_text,k = 100) :
  targetSentencePreprocess = textPreprocess(claim_text)
  tfidfMatrixTarget = vectorizer.transform([targetSentencePreprocess])
  twoMatrixDot = tfidfMatrixTarget.dot(tfIdfMatrix.T)    
  arrayMatrixDot = twoMatrixDot.toarray()
  sortedList = np.argsort(arrayMatrixDot).tolist()[0]
  top10Value = []
  list_df = []
  m = 1
  while len(top10Value) < k :
      evidenceTemp = evidenceList[sortedList[-m]]
      top10Value.append(evidenceTemp)
      m+=1
      temp_dict = {}
      temp_dict['target_sentence'] = claim_text
      temp_dict['evidence_sentence'] = dictEvidence[evidenceTemp]
      list_df.append(temp_dict)
  return pd.DataFrame(list_df) , top10Value


# In[15]:


with open('/Folder dataset/dev-claims.json','r') as f :
    devData = json.load(f)


# In[17]:


from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from statistics import mode
import time
import numpy as np
# you can use it because you don't have pretrain model
cross_encoder_part2 = CrossEncoder('distilroberta-base')

def getLabels(top5Sentence,claim_text) :
  dictClaim = {1 : 'SUPPORTS', 2 : "REFUTES", 3: "NOT_ENOUGH_INFO", 0 : "DISPUTED"}
  listPair = []
  for i in top5Sentence :
    listPair.append([claim_text,i])
  scores = cross_encoder_part2.predict(listPair)
  listLabel = []
  for i in scores.tolist() :
    listLabel.append(i.index(max(i)))
  return dictClaim [mode(listLabel)]

with open('/TF-IDF embedding and vectorize/evidence_list.pkl', 'rb') as handle:
    evidenceList = pickle.load(handle)
devClaimTest = {}
h = 1
start_time = time.time()
for item,val in devData.items() :
  devClaimTest[item] = {}
  devClaimTest[item]['claim_text'] = val['claim_text']
  data_test , evidence_test = top100(val['claim_text'],k=100)
  tested_data = TestedDataset(data_test,maxlen = 60)
  tested_loader = DataLoader(tested_data, batch_size = 100, num_workers = 2)
  evidenceFinal,top5Sentence = predict(net,tested_loader,evidence_test,device)
  devClaimTest[item]['claim_label'] = getLabels(top5Sentence,val['claim_text'])
  devClaimTest[item]['evidences'] = evidenceFinal
  if h % 50 == 0 :
    print(h)
    print(time.time() - start_time)
    print("==================================")
  h += 1


# In[18]:


# this file is used for predicting dev-dataset
with open("devTesting.json", "w") as file:
    json.dump(devClaimTest, file)


# Train The dev dataset to model for testing data

# In[19]:


import time

def train(net, criterion, opti, train_loader, dev_loader, max_eps, device):

    best_acc = 0
    st = time.time()
    for ep in range(max_eps):
        
        net.train()
        for it, (seq, attn_masks,seg_ids, labels) in enumerate(train_loader):
            #Clear gradients
            opti.zero_grad()  
            #Converting these to cuda tensors
            seq, attn_masks, seg_ids ,labels = seq.to(device), attn_masks.to(device),seg_ids.to(device), labels.to(device)

            #Obtaining the logits from the model
            logits = net(seq, attn_masks,seg_ids)

            #Computing loss
            loss = criterion(logits.squeeze(-1), labels.float())

            #Backpropagating the gradients
            loss.backward()

            #Optimization step
            opti.step()
              
            if it % 100 == 0:
                
                acc = get_accuracy_from_logits(logits, labels)
                # TP,FN,FP = get_accuracy_F1_element(logits, labels)
                # F1_result = TP / (TP+0.5*(FP+FN))
                F1_result = calculateTheF1(logits, labels)
                print("Iteration {} of epoch {} complete. Loss: {}; Accuracy: {}; F1-result: {}; ;Time taken (s): {}".format(it, ep, loss.item(), acc,F1_result, (time.time()-st)))
                st = time.time()

        
        dev_acc, dev_loss,F1_result,_ = evaluate(net, criterion, dev_loader, device)
        print("Epoch {} complete! Development Accuracy: {}; Development Loss: {}; Development F1: {}".format(ep, dev_acc, dev_loss,F1_result))
        if F1_result > best_acc:
            print("Best development accuracy improved from {} to {}, saving model...".format(best_acc, F1_result))
            best_acc = F1_result
            torch.save(net.state_dict(), 'sstcls_{}_512_for_predicting.dat'.format(ep))


# In[ ]:


from torch.utils.data import DataLoader
#Creating intsances of training and development dataloaders
train_set = TrainedDataset(filename = 'saved dataset/train-claims-dataset-cosine-100.csv', maxlen = 60)
dev_set = TrainedDataset(filename = 'saved dataset/dev-dataset-cosine-100.csv', maxlen = 60)

train_loader = DataLoader(train_set, batch_size = 100, num_workers = 2)
dev_loader = DataLoader(dev_set, batch_size = 100, num_workers = 2)

num_epoch = 2

#fine-tune the model
train(net, criterion, opti, dev_loader,train_loader, num_epoch,device)


# We can also load trained model

# In[23]:


with open('/Folder dataset/test-claims-unlabelled.json', 'rb') as handle:
    dataTest = json.load(handle)

ClaimTest = {}
for item,val in dataTest.items() :
  ClaimTest[item] = {}
  ClaimTest[item]['claim_text'] = val['claim_text']
  data_test , evidence_test = top100(val['claim_text'])
  tested_data = TestedDataset(data_test,maxlen = 60)
  tested_loader = DataLoader(tested_data, batch_size = 100, num_workers = 2)
  evidenceFinal,top5Sentence = predict(net,tested_loader,evidence_test,device)
  ClaimTest[item]['claim_label'] = getLabels(top5Sentence,val['claim_text'])
  ClaimTest[item]['evidences'] = evidenceFinal


# In[24]:


with open("TestingSubmit.json", "w") as file:
    json.dump(ClaimTest, file)

