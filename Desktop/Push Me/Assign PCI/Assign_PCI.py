"""
This program creates the PCI from the 2 csv files which contain crack data (cilfe) and
pothole data (pfile)
"""

import csv
import os
directory = '/home/christian/Desktop/Rowan CREATEs/March Data/FS@/'

files = os.listdir(directory)
files.sort()


def main(cfile, pfile):
    print(cfile)
    print(pfile)
    f1 = directory + cfile
    f2 = directory + pfile
    street = cfile[:cfile.index('_')]

    
    # Open the first CSV file and count the number of 1.0s in the second column
    with open(f1, 'r') as file1:
        reader1 = csv.reader(file1)
        data1 = list(reader1)
        cracks = 0
        potholes_exclude = []
    
    
    # Open the second CSV file and count the number of 1.0s in the second column
    with open(f2, 'r') as file2:
        reader2 = csv.reader(file2)
        data2 = list(reader2)
        potholes = 0
            
    #Find csvdata 1 not in csvdata 2
    for row in data1:
        if row[1] == '1.0':
            if not any(d[0] == row[0] and d[1] == '1.0' for d in data2):
                cracks += 1
            else:
                potholes_exclude.append(row[0])
    
    for row in data2:
            if row[1] == '1.0' and row[0] not in potholes_exclude:
                potholes += 1
    
    
    # Append the data from both files to separate arrays for each column
    names1 = [row[0] for row in data1]
    values1 = [row[1] for row in data1]
    names2 = [row[0] for row in data2]
    values2 = [row[1] for row in data2]
    total_imgs = len(names1)
    
    # Count the number of images with a 1.0 in both files for the same name
    common_count = 0
    for i in range(total_imgs):
        if names1[i] in names2 and values1[i] == '1.0' and values2[names2.index(names1[i])] == '1.0':
            common_count += 1
    
    # Print the results
    print(street)
    print(f'Cracks: {cracks}')
    print(f'Potholes: {potholes}')
    print(f'C&P: {common_count}')
    print(f'Total Images: {total_imgs}\n')
    

    
    PCI = (100 - (35.147*potholes
                  +12.448*cracks
                  +52.405*common_count)
                   /total_imgs)
    
    
    with open(directory+'PCI_Info.csv', 'a') as rr:
        rr.write(f'{street},{cracks},{potholes},{common_count},{PCI}\n')


count = 0
for y in range(len(files)-1):
    if count % 2 == 0:
        main(files[y], files[y+1])
    count +=1
