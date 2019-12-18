"""
Implementation of a single perceptron using delta rule 

Problem description:
Class labels - given in column 1 as A or B =, to be treated as 1 and 0 respectively 
Compute the error rate i.e no. of misclassified instances using constant learning rate(itat = ita0) and annealing learning rate (itat = ita0/t) for 100 iterations

Format to execute the program:
python calida_perceptron.py --data "D:\Example.tsv" --output "D:\exampleSolution.tsv"

"""

import argparse
import pandas as pd
import csv

#Function to take input from cmd
def userInput():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data" , help="Data file location")
    parser.add_argument("--output", help="Output file location")

    args=parser.parse_args()
    
    inputFile = args.data
    outputFile = args.output

    return inputFile, outputFile


#Function to replace the class labels by boolean values 
def replaceWithBoolean(df):
    total_rows = df.shape[0]
    r,c = df.shape
    print(c)
    #Replacing A=1 & B=0 
    for i in range (total_rows):
        if df[0].values[i]=="A":
            df[0].values[i]=1
        else:
            df[0].values[i]=0


#Function to get output of the perceptron
def findOutputOfPerceptron(w0,w1,w2):
    total_rows = df.shape[0]

    df[3] = w0+w1*df[1]+w2*df[2] #summation of wi*xi
        
    for i in range (total_rows): #Applying activation function to return output i.e >0 returns 1 else returns 0
        if df[3].values[i]>0:
            df[3].values[i]=1
        else:
            df[3].values[i]=0


#Function to find error 
def findError():
    df[4] = df[0]-df[3]     #actual output - perceptron output


#Function to calculate change in weight for contant learning rate 
#For constant learning rate ita = ita0 = 1 
def calculatingUpdatedWeightsForConstantLearningRate(w0,w1,w2):
    ita = 1

    df[5] = df[4]*1              #Calculating E*x0 where x0=1
    df[6] = df[4]*df[1]          #Calculating E*x1 
    df[7] = df[4]*df[2]          #Calculating E*x2
        
    w0 = w0 + ita*sum(df[5])     #Change in weight w0
    w1 = w1 + ita*sum(df[6])     #Change in weight w1
    w2 = w2 + ita*sum(df[7])     #Change in weight w2
    
    return (w0,w1,w2)


#Function to calculate change in weight for annealing learning rate 
#Annealing learning rate ita = ita0/itr, itr is the iterator starting from 1 and ita0 = 1
def calculatingUpdatedWeightsForAnnealingLearningRate(w0,w1,w2,aita):
    df[5] = df[4]*1              #Calculating E*x0 where x0=1
    df[6] = df[4]*df[1]          #Calculating E*x1 
    df[7] = df[4]*df[2]          #Calculating E*x2
   
    w0 = w0 + aita*sum(df[5])    #Change in weight w0
    w1 = w1 + aita*sum(df[6])    #Change in weight w0
    w2 = w2 + aita*sum(df[7])    #Change in weight w0
    
    return (w0,w1,w2)


#Function to find the error for a constant learning rate
def findConstantLearningRateError():
    error = []                  
    itr = 0                      
    w0 = w1 = w2 = 0             
    while (itr<101):             
        findOutputOfPerceptron(w0,w1,w2)
        findError()
        error.append((df[4]!=0).sum())
        w0,w1,w2 = calculatingUpdatedWeightsForConstantLearningRate(w0,w1,w2)
        itr = itr + 1    
    return(error)
    
    

#Function to find the error for annealing learning rate 
def findAnnealingLearningRateError():
    aerror=[]
    ita = 1  
    itr=1
    w0 = w1 = w2 = 0 
    while (itr<=101): 
        aita = ita/itr    
        findOutputOfPerceptron(w0,w1,w2)
        findError()
        aerror.append((df[4]!=0).sum())
        w0,w1,w2 = calculatingUpdatedWeightsForAnnealingLearningRate(w0,w1,w2,aita)
        itr = itr + 1   

    return(aerror)
    
    
#Function to write values to tsv
def writeToTSV(error,aerror,outputFile):
    with open(outputFile, 'w',newline='') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(error)
        tsv_writer.writerow(aerror)


if __name__ =="__main__":

    inputFile,outputFile = userInput()
    df = pd.read_table(inputFile,header=None) 
    replaceWithBoolean(df)
    cerror = findConstantLearningRateError()
    aerror = findAnnealingLearningRateError()
    writeToTSV(cerror,aerror,outputFile)
    





