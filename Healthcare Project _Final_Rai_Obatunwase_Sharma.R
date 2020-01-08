ht.ft = read.csv("HealthcareProject.csv", header = T)
names(ht.ft)
sapply(ht.ft, class)
head(ht.ft)
attach(ht.ft)
library(glmnet)
library(class)
library(MASS)

############################################################################
                          #DATA EXPLORATION
############################################################################
##Scaling the Data
ht.df = cbind(data.frame(scale(ht.ft[, -c(1,4)])),Year,State) #Getting only numerical variables to allow for correlation plot
names(ht.df)

##BoxPlot to compare costs among 4 states
data.for.plot = ht.ft[, c(2,4)]
boxplot(data.for.plot$Healthcare~data.for.plot$State, ylab= "Healthcare Costs", xlab = "State", main= "Health Care Expenditure per State",col = c("Blue", "Blue", "Blue", "Red"))

##Correlation
round(cor(ht.df[,-c(1,10:11)]),2)#Correlation plot
#plot(ht.df) #Scatterplot - there is stong correlation amongst the variables
library(ggplot2)
library(GGally)
ggpairs(ht.df[,-c(10:11)])##Per_Aging_M is strongly correlated with Per_Aging_M and Percentage.of.Aging.population.65.and.over. So, we chose Per_Aging_M
##########################################################################
#PCA - UnSupervised Learning
##########################################################################
##Dimension Reduction
##PRINCIPAL COMPONENT ANALYSIS
library(pls)
head(ht.df)

#Principal Component Analysis(PCA) - Unsupervised Approach
pca = prcomp(ht.df[,-c(1,11:13)], scale. = T)
summary(pca)
pca$rot
#We had to scale again as Year without scaling has a weight of 0.914. After scaling the weight of Year changed to 0.39.
#From the PCA, some of the predictors impact the direction of the response. However, there is no guarantee
#that the predictors will also be the best directions to use for predicting the response. Fitting the linear model might
#show that some of these variables are not statistically significant in predicting the response.

#PRINCIPAL COMPONENTS REGRESSION
pcapcr.fit=pcr(Healthcare~., data=ht.df[,-(11:13)], validation ="CV")
summary(pcapcr.fit)

#getting MSE plot for each 
validationplot(pcapcr.fit, val.type = "MSEP")#selecting the best PC's by plotting Mean sqaure error rate.
#The smallest cross validation error occur at 8. However, using 8 component seem not to be reasonable,it might as well amount
#to performing least square. It suffice to use 5 components.
#which captures 99.43% of the total variation.

#Performing PCR on Training data
trainindex = ht.df[,-c(11:13)]$Year<2014
testdata = ht.df[,-c(11:13)][!trainindex,]

pcr.fit2=pcr(Healthcare~., data=ht.df[,-(11:13)], subset = trainindex, validation ="CV")
validationplot(pcr.fit2 ,val.type="MSEP")# we take M = 5 as the lowest CV error when 5 components are used

pcr.pred=predict(pcr.fit2, testdata, ncomp = 5)
mean((pcr.pred -testdata$Healthcare)^2)
#the test error is competitive (very close) with the result obtained under validation set approach(least square)


##########################################################################
                        #VARIABLE SELECTION
##########################################################################
##(1)shrinkage/Regularization Method (Ridge and Lasso methods)

##RIDGE REGRESSION APPROACH
library(glmnet)
#Glmnet does not use the model formula language so we set up "x" and "y"
x = model.matrix(Healthcare~.-1, data = ht.df) #removing the intercept
y = ht.df$Healthcare #response

#Fitting the ridge regression model
fit.ridge = glmnet(x, y, alpha = 0)#by default alpha is 1, we need to specify alpha = 0 otherwise it wil be treated as Lasso

#Plot the outcome
plot(fit.ridge, xvar = "lambda", label = T)#Increasing lamda (beyond 6) shrinks the coefficient to zero.
#For a relaxed lamda value, the coefficients begin to increase, consequently RSS for the coeeficients are likely to increase. 
#Increasing lamda helps to reduce (shrinks) the size of the coefficients but not make them zero.

plot(fit.ridge, xvar = "dev", label = T)#At deviance = 0.2, 20% of the variability is being explained with slight 
#increase in coefficients. However, at Deviance = 0.8, there is a sudden jump with the coefficient being highly inflated,
#indicating there might be overfitting in that region.

###k-fold Cross Validation (10-fold by default)
cv.ridge = cv.glmnet(x, y, alpha = 0)
cv.ridge
coef(cv.ridge)
plot(cv.ridge)#The red line is the average of the test error

#Getting the required lamda that results in minimum MSE
names(cv.ridge)
cv.ridge$lambda.min #lamda is 0.08985 (minimum value of lamda)
cv.ridge$lambda.1se#This is the value of lamda(0.21) that results in the smallest CV error (maximum error tolerance level)
coef(cv.ridge, s = cv.ridge$lambda.1se)#Getting the coefficients of the estimates for recommended lamda
#Shrinking the coefficients towards zero reduces the variance
#Note: Ridge Regression performs better when response isa function of many predictors. Since P < n for our project, we go with Lasso(easy to interpret)

##################LASSO APPROACH
x = model.matrix(Healthcare~.-1, data = ht.df) #removing the intercept
y = ht.df$Healthcare #response

#Fitting a lasso model using the default alpha = 1
fit.lasso = glmnet(x, y, alpha = 1)#fitting lasso to shrink and select variables
plot(fit.lasso, xvar = "lambda", label = T)#The values on top indicate the number of variables that are non-zero for a given lamda
plot(fit.lasso, xvar = "dev", label = T)
#From 0.8, there is a jump in coefficients indicating the presence of overfitting

###k-fold Cross Validation (10-fold by default)
cv.lasso = cv.glmnet(x, y, alpha = 1)
cv.lasso
plot(cv.lasso)
coef(cv.lasso)#Lasso shrinks some of the variables to zero. Thus, we are left with coefficients corresponding to the best model

#Selecting lamda using the train/validation set
set.seed(222)
trainindex = ht.df$Year<2014
traindata = ht.df[trainindex,]
testdata = ht.df[!trainindex,]
dim(traindata)
dim(testdata)

#Performing cross-validation on training set
cv.train = cv.glmnet(x[trainindex,], y[trainindex], alpha = 1)
coef(cv.train)
plot(cv.train)
cv.train$lambda.min #0.4248
cv.train$lambda.1se#0.4662
coef(cv.train, s = cv.train$lambda.1se)##Sparse model(model with a subset of the variables)

#Estimating Root MSE
lasso.pred = predict(cv.train, x[!trainindex,])
lasso.pred
rmse = sqrt(mean(y[!trainindex] - lasso.pred)^2)
rmse

##(2)Subset Approach
##BEST SUBSET SELECTION APPROACH
library(leaps)
reg.fit = regsubsets(Healthcare~., data = ht.df, nvmax = 10)
reg.summary = summary(reg.fit) 
names(reg.summary)

#Using plot to decide which model to select
par(mfrow = c(2,2))
#RSS
plot(reg.summary$rss,xlab = "Number of variables",type = "b" )
which.min(reg.summary$rss)
points(which.min(reg.summary$rss),reg.summary$rss[which.min(reg.summary$rss)],col="red",cex=2,pch=20)

#Adjusted R^2
plot(reg.summary$adjr2,xlab = "Number of variables",type = "b")
which.max(reg.summary$adjr2)
points(which.max(reg.summary$adjr2),reg.summary$adjr2[which.max(reg.summary$adjr2)],col="red",cex=2,pch=20)

#CP
plot(reg.summary$cp,xlab = "Number of variables",type = "b")
which.min(reg.summary$cp)
points(which.min(reg.summary$cp),reg.summary$cp[which.min(reg.summary$cp)],col="red",cex=2,pch=20)

#BIC
plot(reg.summary$bic,xlab = "Number of variables",type = "b")
which.min(reg.summary$bic)
points(which.min(reg.summary$bic),reg.summary$bic[which.min(reg.summary$bic)],col="red",cex=2,pch=20)


#Plot showing the best predictors, ranked according to Cp, r2, adjr2 and bic
par(mfrow=c(1,2))
plot(reg.fit, scale="Cp")
plot(reg.fit, scale="r2")
plot(reg.fit, scale="adjr2")
plot(reg.fit, scale="bic")

################# CHOOSING AMONG THE THE MODELS #####################
##using the Validation set approach

#Note: For these approaches (Validation and Cross Validation) to yield accurate estimates of the
#test error, only the training set must be used to perform all aspects of
#model-fitting-including variable selection. In other words, the determination of
#which model size is best must be made done only with the training set

trainindex = ht.df$Year<2014
traindata = ht.df[trainindex,]
testdata = ht.df[!trainindex,]
dim(traindata)

regfit.best = regsubsets(Healthcare~., data = ht.df[trainindex,], nvmax=10)#Applying regsubset on the trainingset to perform best subset selection

#To compute the validation set error for the best model of each model size, we make a model matrix from the test data
test.mat = model.matrix(Healthcare~., data = ht.df[!trainindex,])

#Next, we run the loop for each size i to extract the coefficients from regfit.best for the best model
val.errors = rep(NA, 10)
for(i in 1:10){
  coefi = coef(regfit.best,id = i) #running the loop for each size i to extract the coefficients from regfit.best for the best model
  pred = test.mat[,names(coefi)]%*%coefi #multiplying the coefficients with the appropriate columns of the test model matrix to form the predictions
  val.errors[i] = mean((ht.df$Healthcare[!trainindex] - pred)^2) #Estimating the test MSE
}
val.errors

which.min(val.errors)#the best model (Model 6) is the one with least MSE error

#Ploting the validation error
par(mfrow=c(1,1))
plot(sqrt(val.errors), ylab = "Root MSE", ylim = c(0, 1), pch = 20, type = "b")#validation error
points(sqrt(regfit.best$rss[-1]),col="blue", pch = 20, type="b")
points(6, sqrt(val.errors[6]), col = "red", pch = 20, cex = 2)
legend("topright", legend = c("Training","Validation","Best Validation"), col = c("blue","black","red"), pch = 20,cex = .8)

#Comparing the train and full data set
coef(regfit.best, 6)#On train Data Set
coef(reg.fit, 6)#On full Data Set

#The best six-variable on the full dataset has similar set of variables with the train dataset. 
#the train set model has Year, however, the full model has Per_Capita_Personal_Income instead of Year.

#Next, We use K-fold cross validation to choose the model with lowest MSE

##Predictive function
predict.regsubsets = function(object,newdata,id,...){
form = as.formula(object$call[[2]])
mat = model.matrix(form,newdata)
coefi = coef(object,id = id)
mat[,names(coefi)]%*%coefi
}
pred

##Using K-Fold Cross Validation
k = 10
set.seed(22)
folds = sample(1:k, nrow(ht.df), replace = TRUE)
cv.errors = matrix(NA, k, 10, dimnames = list(NULL, paste(1:10)))

for(j in 1:k){
  best.fit = regsubsets(Healthcare~., data = ht.df[folds!=k,], nvmax = 10)#train=folds!=k
  for(i in 1:10){
    pred = predict.regsubsets(best.fit,ht.df[folds==j,], id = i)
    cv.errors[j,i] = mean((ht.df$Healthcare[folds==j] - pred)^2)
  }
}

par(mfrow = c(1,1))

rmse.cv = sqrt(apply(cv.errors, 2 ,mean))
plot(rmse.cv, ylab = "Root MSE", xlab = "Number of Variables", main = "K-Fold Cross Validation", pch = 20, type = "b")
points(which.min(rmse.cv), rmse.cv[which.min(rmse.cv)], col = "red", pch = 20)
#The plot shows that the RootMSE(RSE) doesn't change so much after 6 predictors.
#For the selection purposes, we will use 7-variable model to get the lowest possible test error. 

#The cross-validation selects an 7-variable model. Next is to perform
#best subset selection on the full data set in order to obtain the seven-variable model.

reg.best = regsubsets(Healthcare~., data = ht.df, nvmax = 10)
coef(reg.best, 7) #Choosing the best 7
#For the regression task, we create a linear model choosing the 7 predictors

############################################################################
                          #REGRESSION TASK
############################################################################
##Data partitioning
trainindex = ht.df$Year<2014
traindata = ht.df[trainindex,]
testdata = ht.df[!trainindex,]
dim(traindata)

row.names(traindata)<-c(1:nrow(traindata))
row.names(testdata)<-c(1:nrow(testdata))

#Model Fitting Before variable Selection
colnames(ht.df)
ht.fit3 = lm(Healthcare~., data = ht.df)#fitting the model before variable selection
summary(ht.fit3)
par(mfrow = c(2,2))
plot(ht.fit3)

ht.pred = predict(ht.fit3, newdata = testdata)
mean((testdata$Healthcare-ht.pred)^2) #0.008349

#On UnScaled Data
trainingindex=ht.ft$Year<2014
ht.pred = predict(ht.fit3, newdata = ht.ft[-trainingindex,])
sqrt(mean((ht.ft[-trainingindex,]$Healthcare - ht.pred)^2)) #33658.27


dim(ht.df)
#fiiting the model after variable selection
ht.fit5 = lm(Healthcare~Year+Population+Per_Capita_Personal_Income+Real_Median_Hshd_Income+State, data = traindata)
summary(ht.fit5)
par(mfrow = c(2,2))
plot(ht.fit5)

length(predict(ht.fit5))

#Studentized Residuals
par(mfrow = c(1,1))
plot(predict(ht.fit5),rstudent(ht.fit5),xlab="Fitted Values" , ylab="Studentdized Residual", main="Studentized Residuals vs Fitted Values")
#identify(predict(ht.fit5),rstudent(ht.fit5))
#Press ESC after done 
#As observed in the diagnostic plot, there is a presense of an outlier(19 the observation). 
#This could also be observed in the Studentized Residual plot.

#Validation Set Approach 
ht.pred = predict(ht.fit5, newdata = testdata)
mean((testdata$Healthcare-ht.pred)^2)#Test MSE = 0.054

ht.pred1 = predict(ht.fit5, newdata = traindata)
mean((traindata$Healthcare-ht.pred1)^2)#Train MSE = 0.0016

#Removing the 19th training obeservation
ht.fit55 = lm(Healthcare~Year+Population+Per_Capita_Personal_Income+Real_Median_Hshd_Income+State, data = traindata[-19,])
summary(ht.fit55)
par(mfrow = c(2,2))
plot(ht.fit55)

#Validation Set Approach 
ht.pred2 = predict(ht.fit55, newdata = testdata)
mean((testdata$Healthcare-ht.pred2)^2)#Test MSE = 0.073

ht.pred3 = predict(ht.fit55, newdata = traindata)
mean((traindata$Healthcare-ht.pred3)^2)#Train MSE = 0.0019

#############THE TEST AND TRAINING ERROR ON UNSCALLED DATA for FINAL MODEL############
trainingindex=ht.ft$Year<2014 #Data partioning using the unscaled data(ht.ft)

#Cross Validation Set Approach 
ht.pred = predict(ht.fit5, newdata = ht.ft[-trainingindex,])
sqrt(mean((ht.ft[-trainingindex,]$Healthcare - ht.pred)^2))#Test RSE = $28471.36M

ht.pred1 = predict(ht.fit5, newdata = ht.ft[trainingindex,])
sqrt(mean((ht.ft[trainingindex,]$Healthcare - ht.pred1)^2))#Train RSE = $19521.67M


#Predicting 2019 numbers. 
#Load the 2019 Estimate.csv file
ht.19<-read.csv("2019 Estimates.csv")
ht.pred19 = predict(ht.fit5, newdata =ht.19)
names(ht.pred19) <- c("PA","NJ","NY","IL")
ht.pred19

####LOOCV vs K-Fold
##LOOCV (We dont need to split the data, it splits automatically , n - 1)
#Fit a linear model using glm
#to see if an interaction term gives a better MSE
ht.fit6 = glm(Healthcare~Year+Population+Per_Capita_Personal_Income*Real_Median_Hshd_Income+State, data = ht.df)
summary(ht.fit6)
library(boot)
#Scalled
MSE.LOOCV = cv.glm(ht.df, ht.fit6)$delta[1]
MSE.LOOCV


#Unscalled 
MSE.LOOCV = cv.glm(ht.ft, ht.fit6)$delta[1]
MSE.LOOCV
sqrt(MSE.LOOCV) #0.01226


##K-Fold Cross Validation
  set.seed(1)
  cv.error.10=rep(0,10)
  for (i in 1:10) {
  glm.fit=glm(Healthcare~Year+Population+Per_Capita_Personal_Income+Real_Median_Hshd_Income+State, data = ht.df)
  cv.error.10[i]=cv.glm(ht.df,glm.fit,K=10)$delta[1]  
  }
  cv.error.10
  par(mfrow=c(1,1))
  plot(1:10,cv.error.10,type = "b",col="red", main="K-Fold")
  points(which.min(cv.error.10), cv.error.10[which.min(cv.error.10)], col = "green", pch = 20)
  

##########################################################################
#CLASSIFICATION TASK
#########################################################################

##LOGISTIC REGRESSION
#Cost greater than 50% Quantile is classified as High
summary(ht.ft)
HighLow = as.factor(ifelse(ht.ft$Healthcare>quantile(ht.ft$Healthcare)[3], "High", "Low"))
ht.df2=cbind(ht.df,HighLow)
library(MASS)
ht.df2 = ht.df2[,-1]
head(ht.df2)
attach(ht.df2)

ClassTrainIndex= ht.df2$Year<2014
ClassTrainData= ht.df2[ClassTrainIndex,]
ClassTestData= ht.df2[!ClassTrainIndex,]

glm.fit=glm(HighLow~Year+Population+Per_Capita_Personal_Income+Real_Median_Hshd_Income+State, 
            data = ht.df2,subset=ClassTrainIndex,family = "binomial")#Logistic regression
glm.prob=predict(glm.fit,newdata = ClassTestData,type="response")
glm.pred=ifelse(glm.prob>0.5,"High","Low")
table(glm.pred, HighLow[!ClassTrainIndex])
mean(glm.pred != HighLow[!ClassTrainIndex])#Misclassification Error
mean(glm.pred ==HighLow[!ClassTrainIndex])#Model Accuracy 


#K-Fold CV LOGISTIC
library(caret)
k = 10
set.seed(1)
folds<-createFolds(ht.df2$HighLow)
glm.err = rep(0,10)
for (i in 1:k){
  test<- ht.df2[folds[[i]],]  
  train<- ht.df2[-folds[[i]],]
  glm.fit=glm(HighLow~Year+Population+Per_Capita_Personal_Income+Real_Median_Hshd_Income+State, 
              data = train,family = "binomial")#Logistic regression
  glm.prob=predict(glm.fit,newdata = test,type="response")
  glm.pred=ifelse(glm.prob>0.5,"High","Low")
  glm.err[i]=mean(glm.pred != test$HighLow)#MisclassificationError
}
glm.err
plot(1:10,glm.err, type="b", xlab="Number of Folds",ylab="Missclassification Rate", main="K-fold Cross Validation for Logistic w/ 2 Classes", col="blue")


##LINEAR DISCRIMINANT ANALYSIS (2 levels)
library(MASS)
lda.fit = lda(HighLow~Year+Population+Per_Capita_Personal_Income+Real_Median_Hshd_Income+State, 
              data = ht.df2, subset = ClassTrainIndex)
lda.fit$counts
lda.fit$prior
lda.pred = predict(lda.fit, newdata= ClassTestData)
testclass=ifelse(lda.pred$posterior[,1]>0.5,"High", "Low")
test.Healthcost = HighLow[!ClassTrainIndex]
table(testclass, test.Healthcost)
mean(testclass != test.Healthcost)#Misclassification Error
mean(testclass == test.Healthcost)#Model Accuracy 

##plotting the histogram of LDA function
plot(lda.fit, dimen = 1, type = "b") #there is overlap between high and low 

##LDA Partition Plot
library(klaR)
#partimat(HighLow~., data = ClassTrainData, method = "lda") #classification of each and every observation in the training dataset based on the lda model

##k-Fold Classification for LDA

library(caret)
k = 10
set.seed(111)
folds<-createFolds(ht.df2$HighLow)
lda.err = rep(0,10)
for (i in 1:k){
  test<- ht.df2[folds[[i]],]  
  train<- ht.df2[-folds[[i]],]
  lda.fit=lda(HighLow~Year+Population+Per_Capita_Personal_Income+Real_Median_Hshd_Income+State, data = train)
  lda.pred<- predict(lda.fit,test)
  lda.err[i] <- mean(lda.pred$class != test$HighLow)
}
lda.err
plot(1:10,lda.err, type="b", xlab="Number of Folds",ylab="Missclassification Rate", main="K-fold Cross Validation for LDA w/ 2 Classes", col="blue")



##########################

## K-NEAREST NEIGHBOR(KNN)
library(class)
attach(ht.df2)
Xlag=data.frame(Year,Population,Per_Capita_Personal_Income,Real_Median_Hshd_Income,Per_Aging_M)
Xlag

#K=4
set.seed(1)
knn.pred=knn(Xlag[ClassTrainIndex,],Xlag[!ClassTrainIndex,],HighLow[ClassTrainIndex],k=4)
table(knn.pred,HighLow[!ClassTrainIndex])
mean(knn.pred!=HighLow[!ClassTrainIndex])#MisclassificationError
mean(knn.pred==HighLow[!ClassTrainIndex])#Model Accuracy 

#Misclassification Error at different values of K. 
k.err=rep(0,10)
for (i in 1:10) {
  knn.pred=knn(Xlag[ClassTrainIndex,],Xlag[!ClassTrainIndex,],HighLow[ClassTrainIndex],k=i)
  table(knn.pred,HighLow[!ClassTrainIndex])
  k.err[i]=mean(knn.pred!=HighLow[!ClassTrainIndex])#MisclassificationError
}
k.err

plot(1:10,k.err, type="b", xlab="K",ylab="Missclassification Rate", main="Misclassification errors for K 1 to 10", col="blue")



###############################################
##LINEAR DISCRIMINANT ANALYSIS WITH 3 LEVELS
###############################################
##Creating 3 levels
quantile((ht.df$Healthcare))

Health.cost=ifelse(ht.df$Healthcare <quantile((ht.df$Healthcare))[3],"Low",
       
       ifelse(ht.df$Healthcare >quantile((ht.df$Healthcare))[4],"High","Medium"))

ht.df3=cbind(ht.df,Health.cost)
ht.df3=ht.df3[,-1]
train.x = ht.df3$Year<2014
test.x = ht.df3[!train.x,]
str(ht.df3)

##LDA for 3 classes
lda.fit2 = lda(Health.cost~Year+Population+Per_Capita_Personal_Income+Real_Median_Hshd_Income, 
               data = ht.df3, subset = train.x)
lda.fit2$counts
lda.fit2$prior
lda.pred2 = predict(lda.fit2, newdata= test.x)
test.Healthcost = ht.df3$Health.cost[!train.x]
table(lda.pred2$class, test.Healthcost)
mean(lda.pred2$class != test.Healthcost) #Misclassification Error
mean(lda.pred2$class == test.Healthcost) #Model Accuracy

plot(lda.fit2, col = as.numeric(ht.df3$Health.cost))
plot(lda.fit2, dimen = 1, type = "b") #there is a no overlap between high and low 

### K-Fold Classification
k = 10
set.seed(111)
folds<-createFolds(ht.df3$Health.cost)
lda.err = rep(0,10)
for (i in 1:k){
  test<- ht.df3[folds[[i]],]  
  train<- ht.df3[-folds[[i]],]
  lda.fit=lda(Health.cost~Year+Population+Per_Capita_Personal_Income+Real_Median_Hshd_Income+State, data = train)
  lda.pred<- predict(lda.fit,test)
  lda.err[i] <- mean(lda.pred$class != test$Health.cost)
}
lda.err
plot(1:10,lda.err, type="b", xlab="Number of Folds",ylab="Missclassification Rate", main="K-fold Cross Validation for LDA w/ 3 Classes", col="brown")


############################################################
#Decision Tree
############################################################
##### Classification Tree #######                    
#Unpruned Classification tree
par(mfrow=c(1,1))
library(tree)
ClassTrainIndex= ht.df2$Year<2014
ClassTrainData= ht.df2[ClassTrainIndex,]
ClassTestData= ht.df2[!ClassTrainIndex,]
head(ht.df2)
tree.class = tree(HighLow~., data = ht.df2, subset = ClassTrainIndex)
tree.class

tree.prd = predict(tree.class, ClassTestData, type = "class")
table(tree.prd, ClassTestData$HighLow)
mean(tree.prd != ClassTestData$HighLow) #Misclassification error = 0.25
mean(tree.prd == ClassTestData$HighLow)#prediction accuracy = 0.75

#Pruned Classification tree
#to see if pruning the tree might lead to improved results
set.seed(456)
cv.class = cv.tree(tree.class, FUN = prune.misclass)
cv.class#number of terminal nodes = 3 (size)
names(cv.class)
par(mfrow = c(1,2))
plot(cv.class$size, cv.class$dev, type = "b")
plot(cv.class$k, cv.class$dev, type = "b")

#ploting the pruned tree
par(mfrow=c(1,1))
prune.class = prune.misclass(tree.class, best = 3)
plot(prune.class); text(prune.class, pretty = 0)

#Getting the classification error rate on pruned data
tree.prd2 = predict(prune.class, ClassTestData, type = "class")
table(tree.prd2, ClassTestData$HighLow)
mean(tree.prd2 != ClassTestData$HighLow) #Misclassification error = 0.25
mean(tree.prd2 == ClassTestData$HighLow)#prediction accuracy = 0.75
#Both the pruned and unpruned trees produced the same error rate.

###### Regression Tree #######
set.seed(22)
library(tree)
trainindex = ht.df$Year<2014
traindata = ht.df[trainindex,]
testdata = ht.df[!trainindex,]
dim(traindata)
head(ht.df)
tree.health = tree(Healthcare~., data = ht.df, subset = trainindex)
summary(tree.health)#only 3 variables were used in constructing the tree
#plotting the tree

plot(tree.health); text(tree.health, pretty = 0)

#estimating MSE
y.tree = predict(tree.health, newdata = testdata)
sqrt(mean((y.tree - testdata$Healthcare)^2))#1.0633

#the test root MSE is 1.063326 indicating that the the reg tree model
#leads to test predictions that are around $1.0633m 

##### Bagging ######
library(randomForest)
set.seed(2222)
##Data partitioning
trainindex = ht.df$Year<2014
traindata = ht.df[trainindex,]
testdata = ht.df[!trainindex,]
dim(traindata)
head(ht.df)
row.names(traindata)<-c(1:nrow(traindata))
row.names(testdata)<-c(1:nrow(testdata))

#Unpruned bagging
bag.fit = randomForest(Healthcare~., data = ht.df, subset = trainindex, mtry = 10, importance = TRUE)
bag.fit #All the 10 predictors were considered for each split of the tree

#How well does the model perform on the test set
y.bag = predict(bag.fit, newdata = testdata)
mean((y.bag - testdata$Healthcare)^2)#0.6312
#the test MSE associated with with the bagged is 0.6312 realtive to regression tree 1.063326

##############
#changing the number of tree (prunning) using ntree() argument
#Pruned Bagging
bag.fit2 = randomForest(Healthcare~., data = ht.df, subset = trainindex,
                        mtry = 10, ntree = 20)
bag.fit2
y.bag2 = predict(bag.fit2, newdata = testdata)
mean((y.bag2 - testdata$Healthcare)^2)#0.6128
#The tree prunning the tree resulted in reduction of error from 0.6312(pruned) to 0.6128(unpruned

####Classification for Bagging#####
ht.df2$HighLow = ifelse(HighLow == "Low", 0, 1)
train.id = sample(1:nrow(ht.df2), nrow(ht.df2)*2/3)
train = ht.df2[train.id, ]
test = ht.df2[-train.id, ]
bag.fit3 = randomForest(HighLow~., data = ht.df2, subset = train.id,
                        ntree = 20)
oob.error = double(10)
test.err = double(10)

for(i in 1:10) {
  bag.fit3 = randomForest(HighLow~., data = ht.df2, subset = train.id, mtry =i, ntree =20)
  names(bag.fit3)
  oob.error[i] = bag.fit3$mse[20]
  
  pred = predict(bag.fit3, test)
  test.err[i] = with(test, mean((ht.df2$HighLow - pred)^2))
  test.err  
}
matplot(1:i, cbind(oob.error,test.err),
        col = c("red", "green"), type = "b", ylab = "MSE", xlab = "mtry")
legend("topright", legend = c("OOB", "Test"),pch =19, col = c("red", "green"))

###### Random Forest ######
set.seed(123)
rf.fit = randomForest(Healthcare~., data = ht.df, subset = trainindex, mtry = 6, importance = TRUE)
y.rf = predict(rf.fit, newdata = testdata)
mean((y.rf - testdata$Healthcare)^2)#0.5498
#MSE for random forest is lower than unpruned Bagging but higher than pruned bagging

#Important variable
importance(rf.fit)
varImpPlot(rf.fit)


###### Boosting ########
trainindex = ht.df$Year<2014
traindata = ht.df[trainindex,]
testdata = ht.df[!trainindex,]
set.seed(345)
library(gbm)
boost.fit = gbm(Healthcare~., data = ht.df, distribution = "gaussian", n.trees = 500, interaction.depth = 2)
summary(boost.fit) #Population and State are by far the two most important variables
#getting the MSE
y.boost = predict(boost.fit, newdata = testdata, n.trees = 500)
mean((y.boost - testdata$Healthcare)^2)#0.0111
#boosting yields the smallest error relative to regression, bagging and random forest

####Classification for Boosting ######
ht.df2$HighLow = ifelse(HighLow == "Low", 0, 1)

boost.rf = gbm(HighLow~., data = ht.df2, distribution = "bernoulli", interaction.depth = 4, shrinkage = 0.1, n.trees = 500)
summary(boost.rf)#there is a risk of overfitting
#getting the best number of trees
predmat = predict(boost.rf, newdata = ht.df2, n.trees = 500)
mean(predmat - ht.df2$HighLow)^2 #error is 0.006539531



