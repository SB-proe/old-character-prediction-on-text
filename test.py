
import json


default_loc = "processed_data/"

def generate_char_data(filename):
  sen_list = []
  loc = default_loc + filename
  with open(loc, "r") as f:
    for line in f:
      nl = line.split(',')
      nl[1] = nl[1].rstrip('\n')
      sen_list.append(nl)
  return sen_list

# def test_train_set(sentence_list):
#   i = 0
#   train_set = []
#   test_set = []
#   for line in sentence_list:
#     if i % 10 == 0:
#       test_set.append(line)
#     else:
#       train_set.append(line)
#     i += 1
#   return train_set, test_set

"""
Class that represents a certain character's language modeled with N-Grams.
"""
class XIVBert:
  def __init__(self, name):
    self.name = name
    self.data = generate_char_data(name + ".csv")
    # for x in self.test_set:
    #   x.insert(0, "<s>")
    #   x.insert(0, "<s>")
    #   x.append("</s>")
    # self.train_pipe, self.vocab_pipe = padded_everygram_pipeline(order, self.train_set)
    # self.vocab = Vocabulary(self.vocab_pipe, unk_cutoff=threshold)

urianger = XIVBert("URIANGER")
yshtola = XIVBert("YSHTOLA")
thancred = XIVBert("THANCRED")
alphinaud = XIVBert("ALPHINAUD")
alisaie = XIVBert("ALISAIE")
yda = XIVBert("YDA")
# papalymo = XIVBert("PAPALYMO")
cid = XIVBert("CID")
estinien = XIVBert("ESTINIEN")
# gaius = XIVBert("GAIUS")
grahatia = XIVBert("GRAHATIA")
# nero = XIVBert("NERO")
tataru = XIVBert("TATARU")

characters = [alisaie, alphinaud, cid, estinien, grahatia, tataru, thancred, urianger, yda, yshtola]

import pandas as pd

all_data = []

size = len(characters)

for x in range(len(characters)):
  if x < size:
    for n in characters[x].data:
      all_data.append(n)

pd_data = pd.DataFrame(data=all_data)
print(pd_data)

import numpy as np
import torch
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

## Want BERT instead of distilBERT? Uncomment the following line:
#model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

tokenized = pd_data[1].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape

input_ids = torch.tensor(np.array(padded))
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

features = last_hidden_states[0][:,0,:].numpy()

labels = pd_data[0]
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.10)

#print(train_features)
print(train_labels)

lr_clf = LogisticRegression(multi_class='multinomial', solver='sag', max_iter=10000)
lr_clf.fit(train_features, train_labels)

print(lr_clf.score(test_features, test_labels))

preds = lr_clf.predict(test_features)

#print the tunable parameters (They were not tuned in this example, everything kept as default)
params = lr_clf.get_params()
print(params)

#lr_clf.score(test_features, test_labels)
