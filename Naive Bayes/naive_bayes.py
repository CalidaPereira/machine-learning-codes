
"""
Machine Learning Programming Assignment 4

Author: Calida Pereira 
Date: 15/12/2019
Matriculation No. : 229945

Format to execute the program:

python calida_naive_bayes.py --data "D:\Masters\MachineLearning\Programming Assignments\Assignment 4\Example.tsv" --output "D:\Masters\MachineLearning\Programming Assignments\Assignment 4\exampleSolution.tsv"

python calida_naive_bayes.py --data "D:\Masters\MachineLearning\Programming Assignments\Assignment 4\Gauss2.tsv" --output "D:\Masters\MachineLearning\Programming Assignments\Assignment 4\gaussSolution.tsv"

"""

import argparse
import pandas as pd
import csv
import math

#Function to take input from cmd
def userInput():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data" , help="Data file location")
    parser.add_argument("--output", help="Output file location")

    args=parser.parse_args()
    
    inputFile = args.data
    outputFile = args.output

    return inputFile, outputFile


#Function to drop empty column 
def dropEmptyColumn(df):
    cols = df.shape[1]
    if (str(df[cols-1].values[0]).lower()=="nan"):
           df.drop(cols-1, axis=1, inplace=True)
           
           

#Function to find mean, standard deviation, probability
def findValues(df):
    
    #Find total no. of rows in the dataframe
    rows = df.shape[0]
    
    #Finding class labels
    class_labels = df[0].unique()
    no_of_classes = len(class_labels)
    
    #Finding no. of attributes
    no_of_attr = df.shape[1] - 1

    #Initializing row_values array - will contain the value of the row to be written in the tsv file 
    row_values = [0 for i in range(no_of_classes)]
    
    for k in range(no_of_classes):

        row=[]
        mean=[0 for i in range(no_of_attr)]
        std = [0 for i in range(no_of_attr)]
        
        #Finding prior probability
        df_subtable = df[df[0]==class_labels[k]]
        subtable_rows = df_subtable.shape[0]
        prob_class = subtable_rows/rows

        #Finding mean and standard deviation
        for i in range(no_of_attr):
            mean[i]=df_subtable.iloc[:,i+1].mean()  
            std[i]=df_subtable.iloc[:,i+1].std()
            std[i]=(std[i]**2)
            row.extend([mean[i],std[i]])
       
        row.append(prob_class)
        
        row_values[k]=row
        
    #Returning the calculated values of mean and standard deviation of each class and the no. of classes
    return(row_values,class_labels)
        

#Function to write into the tsv file
def tsvWriter(row_values,no_of_classes):
    with open(outputFile, 'w',newline='') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            #Writing class values
            for k in range(no_of_classes):
                tsv_writer.writerow(row_values[k])
            #Writing misclassifications
            tsv_writer.writerow([row_values[k+1]])
            
      
#Function to find no. of misclassifications 
def findMisclassifications(df):
    rows = df.shape[0]
    #Getting calulated values of the classes
    rv,cv = findValues(df)
    
    #Getting index of the prior probability
    prob_class_index = len(rv[0])-1
    
    #Finding no. of attributes
    no_of_attr = df.shape[1] - 1
    
    #Initializing counter for misclassifications
    counter = 0 

    for i in range(rows):
        
        #Initializing array - numerator - class values 
        #P(X|C)*P(C) 
        pxgc_pc=[]
        
        for k in range(len(cv)):
        
            mc = 1
            vc = 0
            
            #P(X|C) for each attribute
            pxgc = []
            
            for u in range(no_of_attr):
          
                pxgc.append((((1/math.sqrt((2*math.pi*(rv[k][mc]))))* math.exp((-(df[u+1].values[i] - rv[k][vc])**2)/(2*rv[k][mc])))))
                
                #Updating indexes
                mc = mc + 2
                vc = vc + 2
           
            #P(X|C) = P(x1|c)*P(x2|c) 
            likelihood = 1
            for h in pxgc:
                likelihood = likelihood*h
                
               
            pxgc_pc.append(likelihood*rv[k][prob_class_index])

        #Denominator - P(X|C1)*P(C1) + P(X|C2)*P(C2)
        sumval = 0
        for p in pxgc_pc:
            sumval = sumval + p
           
        #Calculating P(C|X) 
        pcgx = []
        for l in pxgc_pc:
            pcgx.append(l/sumval)
            
        #Predicting class label 
        #P(C1|X) > P(C2|X) - class C1 else C2 
        if(pcgx[0]>pcgx[1]):
            predicted = cv[0]
        else:
            predicted = cv[1]
        
        #Incrementing counter when labels are misclassified 
        if (df[0].values[i]!=predicted):
            counter = counter + 1
            
    return(counter)        
            
                
#Main function
if __name__ =="__main__":

    #Taking input from cmd 
    inputFile,outputFile = userInput()
    df = pd.read_table(inputFile,header=None) 
    dropEmptyColumn(df)
    
    #Finding class values - mean, std dev, probability
    rv,nc=findValues(df)
    no_of_classes = len(nc)
    
    #Finding no. of misclassifications
    rv.append(str(findMisclassifications(df)))
    
    #Writing into the tsv 
    tsvWriter(rv,no_of_classes)
        
