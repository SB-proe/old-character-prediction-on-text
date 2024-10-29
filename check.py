
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

print("Start")
for x in characters:
  temp = x.data
  check = [row[1] for row in temp]
  newtrue = []
  for n in check:
    if n not in newtrue:
      newtrue.append(n)
  print(len(check))
  print(len(newtrue))
  new = ""
  for l in newtrue:
    new = new + str(x.name) + "\t" + str(l) + "\n"
  w = open("truedata/" + str(x.name) + "_true.csv", "w")
  w.write(new)
  w.close()
  print("Complete" + str(x.name))