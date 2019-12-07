"""
The goal of this program is to implement ID3 algorithm for decision trees
The last column of the data set is treated as the class label 
This program was written to find the decision tree structure of the car data set and write it into an xml format.

For data set descriptions : http://archive.ics.uci.edu/ml/datasets/Car+Evaluation

Format to execute the program:

python calida_decisiontree.py --data "D:\car.csv" --output "D:\car_output.xml"
python calida_decisiontree.py --data "D:\nursery.csv" --output "D:\nursery_output.xml"

"""

import argparse    
import pandas as pd
import numpy as np
import lxml
from lxml import etree as etree
import math
import time

#Function to take commandline inputs
def userInput():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data" , help="Data file location")
    parser.add_argument("--output", help="Optput file location")

    args=parser.parse_args()
    
    inputFile = args.data
    outputFile = args.output

    return inputFile, outputFile
	
#Function to find the entropy 
def findEntropy(df):
    lastColumn = df.keys()[-1]
    uniqueValues = df[int(lastColumn)].value_counts().keys().tolist()
    countOfUniqueValues = df[int(lastColumn)].value_counts().tolist()
    total = len(df[lastColumn])
    sumOfEntropies = 0
    for i in range (len(countOfUniqueValues)):
        prob = countOfUniqueValues[i]/total
        entropy = -prob*math.log(prob,c)
        sumOfEntropies = sumOfEntropies + entropy
    return sumOfEntropies
	
#Function to find the weighted sum of entropies for an attribute 
def findSumOfWeightedEntropies(df,attribute):
#Finding Proportions
    lastColumn = df.keys()[-1]
    uniqueValuesOfAttr = df[attribute].value_counts().keys().tolist()
    countOfUniqueValues = df[attribute].value_counts().tolist()
    total = len(df[attribute])
    #print(total)
    proportion = [i / total for i in countOfUniqueValues]

    #Finding unique values of target 
    uniqueValuesOfTarget = df[lastColumn].value_counts().keys().tolist()

    #Finding count of each unique value in the attribute for each unique target label 
    index = uniqueValuesOfAttr
    columns = df[lastColumn].value_counts().keys().tolist()
    df_count = pd.DataFrame(index=index, columns=columns)
    df_count = df_count.fillna(0) 
    #df_count

    for i in range(len(df[attribute])):
        #aV = df[attribute].loc[i]
        #tV = df[lastColumn].iloc[i]
        aV = df[attribute].values[i]
        tV = df[lastColumn].values[i]
        
        
        df_count.loc[aV,tV] = int(df_count.loc[aV,tV])+1

    #print(df_count)

    #Finding sum of each unique value and appending it to the df_count dataframe
    df_count.loc[:,'sum'] = df_count.sum(numeric_only=True, axis=1)
    #print(df_count)

    #Calculating entropy of unique values in the attribute 
    df_entcal = pd.DataFrame(index=index, columns=columns)
    df_entcal = df_entcal.fillna(0) 
    #df_entcal

    for p in uniqueValuesOfAttr:
        for q in columns:
            prob = df_count.loc[p,q]/df_count.loc[p,'sum']
            if prob==0:
                df_entcal.loc[p,q] = 0 
            else:
                df_entcal.loc[p,q] = -prob*math.log(prob,c)

    #print(df_entcal)
    df_entcal.loc[:,'sum'] = df_entcal.sum(numeric_only=True, axis=1)
    #print(df_entcal)

    weights =[]
    for k in uniqueValuesOfAttr:
        weights.append(df_count.loc[k,'sum']/total)

    sumOfWeightedEntropies = 0 
    for k in range (len(weights)):
        sumOfWeightedEntropies = sumOfWeightedEntropies - weights[k]*df_entcal.iloc[k,-1]
    #print(sumOfWeightedEntropies)
    return sumOfWeightedEntropies

#Function to find information gain for an attribute 
def findInfoGain(df,attribute,treeEnt):
    weightedEntropies = findSumOfWeightedEntropies(df,attribute)
    infoGain = treeEnt + weightedEntropies
    return infoGain
	
#Function to get the best attribute 
def getBestAttribute(df,treeEnt):
    ig = []
    lastColumn = df.keys()[-1]
    for i in range (lastColumn):
        ig.append(findInfoGain(df,i,treeEnt))
    a = np.argmax(ig)
    #print(ig)
    return(a)

#Recursive function for ID3 to find the next best attribute/ label
def id3(df,targetAttribute,bestAttribute,remainAttr,currentNode):
        #print("BestAttribute:",bestAttribute)
        
        #count no. of unique labels in the target attribute
        from collections import Counter
        cnt = Counter(x for x in df[targetAttribute])
        
        #If there's only one label in the targetAttribute, means it is pure
        if len(cnt)==1:
            entu = findEntropy(df)
            #print("Test",entu)
            
        else:
            # Get unique values of the best Attribute that will be the edges
            vals = df[bestAttribute].value_counts().keys().tolist()
            
            #Create sub data frames
            for i in vals:
                #print(i)
                df_sub = df.loc[df[bestAttribute]==i]
               
                subEntropy = findEntropy(df_sub)
                #print(subEntropy)

                
                if(subEntropy == 0):
                    node = etree.SubElement(currentNode, "node",entropy="0.0",feature="att{val}".format(val=bestAttribute),value=i)
                    node.text=df_sub.iloc[0,-1]
                else:
                    bestA = getBestAttribute(df_sub,subEntropy)
                    remainingAttributes = [i for i in remainAttr if i != bestA]
                    #print("Next best attribute:",bestA)
                    node = etree.SubElement(currentNode, "node",entropy=str(subEntropy),feature="att{val}".format(val=bestAttribute),value=i)
                    id3(df_sub,cols-1,bestA,remainingAttributes,node)
                    
                    
if __name__ =="__main__":

    #start = time.perf_counter()
    
    #Commandline input from user
    inputFile, outputFile = userInput()

    df_car = pd.read_csv(inputFile,header=None)

    #Finding no. of class labels - this value is also used to compute entropy 
    rows,cols = df_car.shape
    c = len(df_car[cols-1].value_counts().keys()) 

    # Find Tree entropy 
    treeEntropy = findEntropy(df_car)

    #Adding tree entropy to xml - <tree>
    root = etree.Element("tree", entropy=str(treeEntropy))  

    #Finding the best attribute to start the decision tree
    bestAttr = getBestAttribute(df_car,treeEntropy)

    #Find remaining attributes - attributes that 
    attributeNames = df_car.keys().tolist()
    attributeNames.remove(cols-1) #Removing the target column
    remainingAttributes = [i for i in attributeNames if i != bestAttr] #Removing the attribute chosen as the root node
    
    id3(df_car,cols-1,bestAttr,remainingAttributes,root)

    with open(outputFile, 'wb') as doc:
        doc.write(etree.tostring(root, pretty_print = True))
        
    #end = time.perf_counter()
    #print ("Time taken : %.2gs" % (end-start))

