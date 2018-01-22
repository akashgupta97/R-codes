###############################################################
#                                                             #
#            RandomForest model for Regression                #
#                                                             #
###############################################################
#                                                             #
# Credit: Dr. Prashant Singh Rana                             #
# Email : psrana@gmail.com                                    #
# Web   : www.psrana.com                                      #
#                                                             #
###############################################################
#                                                             #
# Train and Test randomForest model for Regression            #
#                                                             #
# This script do the following:                               #
# 1. Load the Data                                            #
# 2. Partition the data into Train/Test set                   #
# 3. Train the randomForest Model                             #
# 4. Test                                                     #
# 5. Evaluate on : Correlation, Regression, RMSE, Accuracy.   # 
# 6. Finally Saving the results.                              #
#                                                             #
###############################################################


#--------------------------------------------------------------
# Step 0: Start; Getting the starting time
#--------------------------------------------------------------
cat("\nRandom Forest")
startTime = proc.time()[3]



#--------------------------------------------------------------
# Step 1: Include Library
#--------------------------------------------------------------
library(randomForest,quietly = TRUE)


#--------------------------------------------------------------
# Step 2: Variable Declaration
#--------------------------------------------------------------
modelName <- "randomForest"
InputDataFileName="regressionDataSet.csv"
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
model   <- randomForest(formula, trainDataset, ntree=500, mtry=2)


#--------------------------------------------------------------
# Step 10: Prediction (Testing)
#--------------------------------------------------------------
Predicted <- predict(model, testDataset)


#--------------------------------------------------------------
# Step 11: Extracting Actual
#--------------------------------------------------------------
Actual <- as.double(unlist(testDataset[target]))


#--------------------------------------------------------------
# Step 12: Model Evaluation
#--------------------------------------------------------------

# Step 12.1: Correlation
r <- round(cor(Actual,Predicted ),2)

# Step 12.2: RSquare
R <- round(r * r, 2)

# Step 12.3: RMSE
rmse <- round(mean(abs(Actual-Predicted)),2)

# Step 12.4: Accuracy
accuracy <- round(mean(abs(Actual-Predicted) <=1),4) *100

# Step 12.5: Total Time
totalTime = proc.time()[3] - startTime

# Step 12.6: Scatter Plot
png(filename=paste(modelName,"-ScatterPlot.png",sep=''))
plot(Actual,Predicted,main=paste("Actual Vs Predicted\n",modelName),xlab="Predicted", ylab="Actual")#, pch=19)
abline(lm(Actual ~ Predicted,),col="White") 
msg<-dev.off()

# Step 12.7: Save evaluation resut 
result <- data.frame(modelName,r,R,rmse,accuracy, totalTime)[1:1,]


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



#--------------------------------------------------------------
#                           END 
#--------------------------------------------------------------
