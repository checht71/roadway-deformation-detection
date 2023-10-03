#Road Ranking
import csv

with open('PCI_Info.csv','r') as file:
    reader = csv.reader(file)
    data = list(reader)
print(data)