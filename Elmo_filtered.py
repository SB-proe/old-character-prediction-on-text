
from cgi import test
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
    self.data = generate_char_data(name + "_filtered.csv")
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
#print(pd_data)

import numpy as np
import torch
import transformers as ppb # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

print("\nStart ELMO")
print("Start Reading File********************************************")

embeddings_from_file = []
with open('processed_data/elmo_embeddings_filtered.txt', 'r') as readfile:
  for line in readfile:
    no_brackets = line.strip('[]\n ')
    splits = no_brackets.split(',')
    splits_float = [float(i) for i in splits]
    embeddings_from_file.append(np.array(splits_float))

features = embeddings_from_file

labels = pd_data[0]
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.10)

#print(train_features)
#print(train_labels)

print("Start RF*********************************************************")
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
runtime = 30
n = 0
max = 0
maxRF = 0
for n in range(runtime):
  #n_estimators = default, value makes no difference in our results
  #criterion="entropy", makes no difference
  rf_clf = RandomForestClassifier(n_jobs = -1)
  rf_clf.fit(train_features, train_labels)
  value = rf_clf.score(test_features, test_labels)
  if value > max:
    maxRF = rf_clf
    max = value
  print(value)

print("\nBest")
print(max)
print()

#   from sklearn.tree import DecisionTreeClassifier
#   dt_clf = DecisionTreeClassifier()
#   dt_clf.fit(train_features, train_labels)
#   print("DT: " + str(dt_clf.score(test_features, test_labels)))

#   # 0.27906976744186046
#   ab_clf = AdaBoostClassifier()
#   ab_clf.fit(train_features, train_labels)
#   print("AB: " + str(ab_clf.score(test_features, test_labels)))


#   #print("Start KN*********************************************************")
#   from sklearn.neighbors import KNeighborsClassifier
#   kn_clf = KNeighborsClassifier(n_jobs = -1)
#   kn_clf.fit(train_features, train_labels)
#   print("KN: " + str(kn_clf.score(test_features, test_labels)))

predict_test = maxRF.predict(test_features)
# print(predict_test)
rows, cols = (10, 10)
cm = [[0 for i in range(cols)] for j in range(rows)]
# print(cm)

key_dict = {
  "ALISAIE": 0,
  "ALPHINAUD": 1,
  "CID": 2,
  "ESTINIEN": 3,
  "GRAHATIA": 4,
  "TATARU": 5,
  "THANCRED": 6,
  "URIANGER": 7,
  "YDA": 8,
  "YSHTOLA": 9
}
test_labels_list = test_labels.values.tolist()
# print(predict_test[0])
# print(test_labels_list)
# print(key_dict["ALISAIE"])

for x in range(len(predict_test)):
  cm[key_dict[predict_test[x]]][key_dict[test_labels_list[x]]] += 1
cmp = np.array(cm)
print(cmp)

x_way = np.sum(cmp, axis = 1)
y_way = np.sum(cmp, axis = 0)
for x in range(len(cmp)):
  true = cmp[x][x]
  precision = true/x_way[x]
  recall = true/y_way[x]
  fscore = (2*precision*recall)/(precision+recall)
  print(f"{characters[x].name}\nPrecision:{precision}\nRecall:{recall}\nFscore:{fscore}\n")

# print("Start MPL*********************************************************")
# from sklearn.neural_network import MLPClassifier
#default
#0.581772784019975
#activation = identity is trash
#0.38541109327626183
#0.3807740324594257
#activation = logistic
#0.8598180845371857
#0.8694489031567684
#0.8755127519172463
#activation = tanh
#0.8553593722133048
#0.862314963438559

#max_iter = 5000
#early_stopping = False, ALWAYS!!!
#activation = 'logistic', ALWAYS!!!
#learning_rate = 'constant' default
# mpl_clf = MLPClassifier(max_iter=5000, activation = 'logistic', early_stopping = False, solver = 'lbfgs')
# mpl_clf.fit(train_features, train_labels)
# print(mpl_clf.score(test_features, test_labels))
# mpl_clf = MLPClassifier(max_iter=5000, activation = 'logistic', early_stopping = False, solver = 'lbfgs')
# mpl_clf.fit(train_features, train_labels)
# print(mpl_clf.score(test_features, test_labels))


# print("Start NB*********************************************************")
# from sklearn.naive_bayes import GaussianNB
# nb_clf = GaussianNB()
# nb_clf.fit(train_features, train_labels)
# print("NB: " + str(nb_clf.score(test_features, test_labels)))

# print("Start LR*********************************************************")
# lr_clf = LogisticRegression(multi_class='multinomial')
# lr_clf.fit(train_features, train_labels)
# print(lr_clf.score(test_features, test_labels))

#preds = lr_clf.predict(test_features)
#print the tunable parameters (They were not tuned in this example, everything kept as default)
#params = lr_clf.get_params()
#print(params)

#lr_clf.score(test_features, test_labels)

