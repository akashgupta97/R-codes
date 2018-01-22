###############################################################
#                                                             #
#            SVM for Classification                           #
#                                                             #
###############################################################
#                                                             #
# Credit: Dr. Prashant Singh Rana                             #
# Email : psrana@gmail.com                                    #
# Web   : www.psrana.com                                      #
#                                                             #
###############################################################
#                                                             #
# Train and Test SVM model for Classification                 #
#                                                             #
# This script do the following:                               #
# 1. Load the Data                                            #
# 2. Partition the data into Train/Test set                   #
# 3. Train the Decision Tree Model                            #
# 4. Test                                                     #
# 5. Evaluate on : Accuracy.                                  # 
# 6. Finally Saving the results.                              #
#                                                             #
###############################################################


#--------------------------------------------------------------
# Step 0: Start; Getting the starting time
#--------------------------------------------------------------
cat("\nSVM\n")
startTime = proc.time()[3]


#--------------------------------------------------------------
# Step 1: Include Library
#--------------------------------------------------------------
library(kernlab)


#--------------------------------------------------------------
# Step 2: Variable Declaration
#--------------------------------------------------------------
modelName <- "svm"
InputDataFileName="classificationDataSetMultiClass.csv"
training = 50      # Defining Training Percentage; Testing = 100 - Training

#--------------------------------------------------------------
# Step 3: Data Loading
#--------------------------------------------------------------
dataset <- read.csv(InputDataFileName)      # Read the datafile
dataset <- dataset[sample(nrow(dataset)),]  # Shuffle the data row wise.


#--------------------------------------------------------------
# Step 4: Count total number of observations/rows.
#--------------------------------------------------------------
totalDataset <- nrow(dataset)


#--------------------------------------------------------------
# Step 5: Choose Target variable
#--------------------------------------------------------------
target  <- names(dataset)[1]   # i.e. RMSD


#--------------------------------------------------------------
# Step 6: Choose inputs Variables
#--------------------------------------------------------------
inputs <- setdiff(names(dataset),target)

#Feature Selection
#n=4
#inputs <-sample(inputs, n)



#--------------------------------------------------------------
# Step 7: Select Training Data Set
#--------------------------------------------------------------
trainDataset <- dataset[1:(totalDataset * training/100),c(inputs, target)]

#--------------------------------------------------------------
# Step 8: Select Testing Data Set
#--------------------------------------------------------------
testDataset <- dataset[(totalDataset * training/100):totalDataset,c(inputs, target)]


#--------------------------------------------------------------
# Step 9: Model Building (Training)
#--------------------------------------------------------------
formula <- as.formula(paste(target, "~", paste(c(inputs), collapse = "+")))
model  <- ksvm(formula, trainDataset, kernel="rbfdot", prob.model=TRUE)

#--------------------------------------------------------------
# Step 10: Prediction (Testing)
#--------------------------------------------------------------
Predicted <- round(predict(model, testDataset))


#--------------------------------------------------------------
# Step 11: Extracting Actual
#--------------------------------------------------------------
Actual <- as.double(unlist(testDataset[target]))


#--------------------------------------------------------------
# Step 12: Model Evaluation
#--------------------------------------------------------------

# Step 12.1: Accuracy
accuracy <- round(mean(Actual==Predicted) *100,2)

# Step 12.2: Total Time
totalTime = proc.time()[3] - startTime

# Step 12.3: Save evaluation resut 
result <- data.frame(modelName,accuracy, totalTime)[1:1,]


#--------------------------------------------------------------
# Step 13: Writing to file
#--------------------------------------------------------------

# Step 13.1: Writing to file (evaluation result)
write.csv(result, file=paste(modelName,"-Evaluation-Result.csv",sep=''), row.names=FALSE)

# Step 13.2: Writing to file (Actual and Predicted)
write.csv(data.frame(Actual,Predicted), file=paste(modelName,"-ActualPredicted-Result.csv",sep=''), row.names=FALSE)



#--------------------------------------------------------------
# Step 14: Saving the Model
#--------------------------------------------------------------
save.image(file=paste(modelName,"-Model.RData",sep=''))


cat("\nDone")
cat("\nTotal Time Taken: ", totalTime," sec")


#--------------------------------------------------------------
#                           END 
#--------------------------------------------------------------


