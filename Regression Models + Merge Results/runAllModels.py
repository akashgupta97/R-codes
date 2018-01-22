#-------------------------------------------------------------
#                   Run Models and Merge Results
#-------------------------------------------------------------
# Python version: 2.7
#-------------------------------------------------------------


#-------------------------------------------------------------
# Step 1: Library inclusion                             
#-------------------------------------------------------------
import os
import glob


#-------------------------------------------------------------
# Step 2: Variable Declaration
#-------------------------------------------------------------
resutlFileName="finalResult.csv"


#-------------------------------------------------------------
# Step 3: Model Execution
#-------------------------------------------------------------

# Decision Tree
os.system("Rscript decisionTree.R")

# Linear Model
os.system("Rscript linearModel.R")

# Neural Network
os.system("Rscript neuralNetwork.R")

# Random Forest
os.system("Rscript randomForest.R")


#-------------------------------------------------------------
# Step 4: Merging Result 
#-------------------------------------------------------------

listOfResultFiles = glob.glob('*Evaluation-Result.csv')
fwp=open(resutlFileName,"w")
fwp.write("Model,r,R,RMSE,Acc,TimeTaken\n")

for f in listOfResultFiles:
    i=1    
    for fp in open(f):
        if i==1:
            i=i+1
            continue
        fwp.write(fp)
        break

fwp.close()
print ("Done")
print ("Result is save in " + resutlFileName + "\n")


