
####################################################################################################################

####################################################################################################################

####################################################################################################################

#############################This program was created by Davit Khachatryan##########################################
######©2017-2018 by Davit Khachatryan.  All rights reserved. No part of this document may  be reproduced or#########
######transmitted in any form or by any means, electronic, mechanical, photocopying, recording or otherwise#########
############################without prior written permission of Davit Khachatryan###################################

####################################################################################################################

####################################################################################################################

####################################################################################################################

library(gmodels)
library(Hmisc)
library(e1071) #The package that supports Naive Bayes. You have to install it first before loading with 
#the "library" command.

#START OF DATA IMPORT

#update the path below to point to the directory and name of your data in *.csv format  

mydata=read.csv("C:/Users/tmai1/Documents/My RStudio/Rawdata/Mushroom Data 2019.csv")
str(mydata)

#END OF DATA IMPORT


#START OF VARIABLE REDEFINITION

mydata$myresponse=mydata$Status #Substitute "Flight.Status" with the name of your response variable
mydata$Status=NULL #Substitute "Response" with the name of your response variable

str(mydata)

#Comment out the statement below ONLY IF your outcome variable is read into RStudio as
#factor. Otherwise, un-comment it and run it.

#mydata$myresponse=as.factor(mydata$myresponse) 

#END OF VARIABLE REDEFINITION


#The statements below remove all the variables that will not be passed to the naive bayes algorithm
#as predictors. If no such redundant variables exist in your dataset, then the statements
#in the "REDUNDANT VARIABLE REMOVAL" section should be deleted or commented out.

#START OF REDUNDANT VARIABLE REMOVAL


#Add as many statements as needed similar to above

#END OF REDUNDANT VARIABLE REMOVAL

str(mydata)

#START OF VARIABLE TRANSFORMATION
#mydata$Weather=as.factor(mydata$Weather)
#mydata$DAY_WEEK=as.factor(mydata$DAY_WEEK)
#Add statements similar to above as needed. All variables that will be used for modeling in this macro should be factors.

#END OF VARIABLE TRANSFORMATION

#############################################################################################
#####################################ATTENTION###############################################
#############################################################################################

#######################IF THE ABOVE MODIFICATIONS ARE MADE CORRECTLY,########################
####AT THIS POINT "MYDATA" DATA FRAME SHOULD CONTAIN ONLY THE PREDICTORS AND THE OUTCOME.#### 
####IN CASE IT CONTAINS ANYTHING MORE OR LESS, THE CODE BELOW WILL NOT FUNCTION PROPERLY.####
#############################################################################################


str(mydata)
summary(mydata)

#############################################################################################
########################DO NOT MODIFY BEYOND THIS POINT######################################
#############################################################################################

#START DATA BREAKDOWN FOR HOLDOUT METHOD

numpredictors=dim(mydata)[2]-1

numfac=0

for (i in 1:numpredictors) {
  if ((is.factor(mydata[,i]))){
    numfac=numfac+1} 
}

nobs=dim(mydata)[1]
set.seed(1) #sets the seed for random sampling

prop = prop.table(table(mydata$myresponse))
length.vector = round(0.8*nobs*prop)
train_size=sum(length.vector)
test_size=nobs-train_size
class.names = as.data.frame(prop)[,1]
numb.class = length(class.names)
resample=1

#The 'while' conditional construct below breaks the data into testing(20%) and training(80%) sets assuring that the unique levels
#of each of the categorical variables is the same in mydata, testing, and training sets. If for a particular partition
#those levels do not match, then RStudio continues to perform 80-20 random splits untill such partition is found.


while (resample==1) {
  
  train_index = c()
  
  for(i in 1:numb.class){
    index_temp = which(mydata$myresponse==class.names[i])
    train_index_temp = sample(index_temp, length.vector[i], replace = F)
    train_index = c(train_index, train_index_temp)
  }
  
  mydata_train=mydata[train_index,] #randomly select the data for training set using the row numbers generated above
  mydata_test=mydata[-train_index,]#everything not in the training set should go into testing set
  
  right_fac=0 #denotes the number of factors with "right" distributions (i.e. - the unique levels match across mydata, test, and train data sets)
  
  for (i in 1:numpredictors) {
    if (is.factor(mydata_train[,i])) {
      if (setequal(intersect(as.vector(unique(mydata_train[,i])), as.vector(unique(mydata_test[,i]))),as.vector(unique(mydata[,i])))==TRUE)
        right_fac=right_fac+1
    }
  }
  
  if (right_fac==numfac) (resample=0) else (resample=1)
  
}  

test_predictors=mydata_test
test_predictors$myresponse=NULL
myresponse_test=as.data.frame(mydata_test$myresponse)
colnames(myresponse_test)="myresponse"
str(test_predictors)
str(myresponse_test)

dim(mydata_test) #confirms that testing data has only 20% of observations
dim(mydata_train) #confirms that training data has 80% of observations

#END DATA BREAKDOWN FOR HOLDOUT METHOD


#START NAIVE BAYES

model=naiveBayes(myresponse ~ .,data=mydata_train)
pred=predict(model, test_predictors)
tbl=as.data.frame(table(myresponse_test$myresponse,pred))
percent_correct=round(100*sum(tbl[which(tbl$Var1==tbl$pred),3])/dim(mydata_test)[1],2)

#END NAIVE BAYES

#START BENCHMARKING COMPARISON

prop_train = as.data.frame(prop.table(table(mydata_train$myresponse)))
prop_train=prop_train[order(-prop_train$Freq),]
dominant_class=prop_train[1,1]
test_benchmark=mydata_test
test_benchmark$simple_classification=as.character(dominant_class)
percent_correct_simple=round(100*sum(test_benchmark$simple_classification==test_benchmark$myresponse)/dim(test_benchmark)[1],2)

print(paste("Percentage of Correct Classifications for Naive Bayes is:",percent_correct, "percent")) 
print(paste("Percentage of Correct Classifications for the Benchmark Classification is:",percent_correct_simple, "percent"))             

Confusion_Matrix=CrossTable(myresponse_test$myresponse,pred, dnn=c("True Class","Predicted Class"), prop.chisq=F,prop.t=F, prop.c=F, prop.r=F)
#END BENCHMARKING COMPARISON

#Start: Putting together the test data with the classifications

for_export=cbind(mydata_test,pred)
for_export$Naive_Bayes_Classification=for_export$pred
for_export$pred=NULL

View(for_export)

#End: Putting together the test data with the classifications

#####################################################################################################  
#If you have truly new records for which you do not know their true classifications,
#but would like to predict using Naive Bayes, you will need to 
#make sure that those records are stored in a data frame called "fresh", and ensure that the mentioned
#data frame has all the predictors that were in your testing set and nothing more. The names of the predictors
#should be exactly the same as the names of the predictors in "mydata".
#Data frame "fresh" needs to have 1 less column than the testing set (since there is a column called "myresponse" in your testing 
#set which won't be present in "fresh"). If the above noted specifications hold, then to generate the classifications
#for the records contained in "fresh" all you have to do is to consider the code below. 
#The resultant classifications will be stored in the last column in data frame called "table_with_classifications".
#####################################################################################################  

#model.full=naiveBayes(myresponse ~ .,data=mydata)
#fresh_classifications=predict(model.full, fresh)
#table_with_classifications=cbind(fresh,fresh_classifications)

#############################################################################################
##############################THIS IS THE END OF THE MACRO###################################
#############################################################################################
