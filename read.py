import json

f = open("output.txt", "r")
print("Start")
line = f.readline()
new = ""
name = "MYSTERYVOICE"
name2 = "GRAHATIA"
while line:
    trim = line.split('\t')
    if trim[1] == name or trim[1] == name2:
        new += str(trim[0] + "\t" + str(trim[1]) + "\t" + str(trim[2]))
    line = f.readline()

w = open(str(name) + ".txt", "w")
w.write(new)
f.close()
w.close()
print("Complete")