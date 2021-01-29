#Machine Learning Toolbox

#supervised learning (predictive learning): classification & regression
library(ggplot2)
library(caret)
data(diamonds)
str(diamonds)


#What is the advantage of using a train/test split rather than just validating your model in-sample on the training set?
#It gives you an estimate of how well your model performs on new data.

#set a random seed so that your work is reproducible and you get the same random split each time you run your script
set.seed(42)

#use the sample() function to shuffle the row indices of the diamonds dataset. You can later use these indices to reorder the dataset.
rows <- sample(nrow(diamonds))

#use this random vector to reorder the diamonds dataset:
shuffled_diamonds <- diamonds[rows, ]

#split the first 80% of it into a training set, and the last 20% into a test set
split <- round(nrow(diamonds) * .80)

#use this point to break off the first 80% of the dataset as a training set
train <- diamonds[1:split, ]

#use that same point to determine the test set
test <- diamonds[(split + 1):nrow(diamonds), ]

#Fit an lm() model called model to predict price using all other variables as covariates. Be sure to use the training set, train.
# Fit lm model on train: model
model <- lm(price ~ ., train)

# Predict on test: p
p <- predict(model, test)

#Calculate the error between the predictions on the test set and the actual diamond prices in the test set. Call this error
error = p - test$price

# Calculate RMSE( Computing the error on the training set is risky because the model may overfit the data used to train it.)
sqrt(mean(error^2))

####CROSSVALIDATION
#As you saw in the video, a better approach to validating models is to use multiple systematic test sets, rather than a single random train/test split
#If all of your estimates give similar outputs, you can be more certain of the model's accuracy. If your estimates give different outputs, that tells you the model does not perform consistently and suggests a problem with it.
library(caret)
#Fit a linear regression to model price using all other variables in the diamonds dataset as predictors. Use the train() function and 10-fold cross-validation. 

model <- train(
  price~ ., 
  diamonds,
  method = "lm",
  trControl = trainControl(
    method = "cv",   #cv je crossvalidation
    number = 10,     #kreira 10 datasets
    verboseIter = TRUE)
)

# Print model to console
model

#or repeated cross-validation
model <- train(
  price~ ., 
  diamonds,
  method = "lm",
  trControl = trainControl(
    method = "cv",   #cv je crossvalidation
    number = 5,     #kreira 10 datasets
    repeats = 5,
    verboseIter = TRUE
  )
)

model

## Predict on full  dataset with caret
predict(model, diamonds)


#######Logistic Regression
dir()
load("~/Desktop/machine_learning/Sonar.rdata")
ls()
str(sonar_test)
#Get the number of observations (rows) in Sonar, assigning to n_obs
n_obs <- nrow(sonar_test)
#Shuffle the row indices of Sonar and store the result in permuted_rows
permuted_rows <- sample(n_obs)
#Use permuted_rows to randomly reorder the rows of Sonar, saving as Sonar_shuffled.
Sonar_shuffled <- sonar_test[permuted_rows, ]
#Identify the proper row to split on for a 60/40 split. Store this row number as split.
split <- round(n_obs * .60)
#Save the first 60% of Sonar_shuffled as a training set.
train <- Sonar_shuffled[1:split, ]
#Save the last 40% of Sonar_shuffled as the test set.
test <- Sonar_shuffled[(split + 1):n_obs, ]
# fit a logistic regression model to your training set using the glm() function. glm() is a more advanced version of lm() that allows for more varied types of regression models, aside from plain vanilla ordinary least squares regression
#Fit a logistic regression called model to predict Class using all other variables as predictors. Use the training set for Sonar
#Predict on the test set using that model. Call the result p like you've done before.
# Fit glm model: model
model <- glm(Class~ ., family = "binomial", train)

# Predict on test: p
p <- predict(model, test, type = "response")
p

#confusion matrix
#confusion matrix is a very useful tool for calibrating the output of a model and examining all possible outcomes of your predictions (true positive, true negative, false positive, false negative)
#"cut" your predicted probabilities at a given threshold to turn probabilities into a factor of class predictions
#Use ifelse() to create a character vector, m_or_r that is the positive class, "M", when p is greater than 0.5, and the negative class, "R", otherwise
m_or_r <- ifelse(p > 0.5, "M", "R") #50% treshold, M stands for mines (positive) and R for rocks(negative)
# Convert to factor: p_class
p_class <- factor(m_or_r, levels = levels(test[["Class"]]))
# Create confusion matrix
confusionMatrix(p_class, test[["Class"]]) #detailed summary of your model's accuracy
#givesv true positive rate (or sensitivity) and true negative rate (or specificity) 

#playing with treshold values (higher the value is the less of the items are predicted in this case mines - zbroj prvog reda)
#use a probability threshold of 0.90 to get fewer predicted mines, but with greater confidence in each prediction
# If p exceeds threshold of 0.9, M else R: m_or_r
m_or_r <- ifelse(p > 0.9, "M", "R")

# Convert to factor: p_class
p_class <- factor(m_or_r, levels = levels(test[["Class"]]))

# Create confusion matrix
confusionMatrix(p_class, test[["Class"]])

#ovo je pretesko svaki put mijenjati p value- bolje koristitit ROC
#ROC curves let you evaluate how good a model is, without worry about calibrating its probabilities.
library(caTools)
# Predict on test: p
p <- predict(model, test, type = "response")

# Make ROC curve
colAUC(p, test[["Class"]], plotROC = TRUE)

#caculate AUC (1-super, 0.5 random, the best > 0.8 but 0.7 is acceptable)
# Create trainControl object: myControl
myControl <- trainControl(
  method = "cv",
  number = 10,
  summaryFunction = twoClassSummary, #ovo smo promijenili
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE
)

#Now that you have a custom trainControl object, it's easy to fit caret models that use AUC rather than accuracy to tune and evaluate the model. 
#You can just pass your custom trainControl object to the train() function via the trControl argument
# Train glm with custom trainControl: model
model <- train(
  Class ~ ., 
  Sonar, 
  method = "glm",
  trControl = myControl
)

# Print model to console
model

########Random forests (A random forest is a more flexible model than a linear model, but just as easy to fit)
#et's try one out on the wine quality dataset, where the goal is to predict the human-evaluated quality of a batch of wine, given some of the machine-measured chemical and physical properties of that bat
library(ranger)
wine <- readRDS("~/Desktop/machine_learning/wine_100.rds")
str(wine)
#Fitting a random forest model is exactly the same as fitting a generalized linear regression model, as you did in the previous chapter. 
#You simply change the method argument in the train function to be "ranger". The ranger package is a rewrite of R's classic randomForest package and fits models much faster, but gives almost exactly the same results. 
#We suggest that all beginners use the ranger package for random forest modeling
# Fit random forest: model
model_wine <- train(
  quality~ ., #quality of the wine is the response variable
  tuneLength = 1,
  data = wine, 
  method = "ranger", #ranger is random Forest package
  trControl = trainControl(
    method = "cv", 
    number = 5, 
    verboseIter = TRUE))
# Print model to console
model_wine

#tuning tune lenght
# Fit random forest: model
model <- train(
  quality~ .,
  tuneLength = 3,
  data = wine, 
  method = "ranger",
  trControl = trainControl(
    method = "cv", 
    number = 5, 
    verboseIter = TRUE
  )
)
model
# Plot model
plot(model)

#Advantages of a custom tuning grid
#tuneGrid=more fine-grained control over the tuning parameters that are explored
# Define the tuning grid: tuneGrid
tuneGrid <- data.frame(
  .mtry = c(2,3,7),
  .splitrule = "variance",
  .min.node.size = 5
)

# From previous step
tuneGrid <- data.frame(
  .mtry = c(2, 3, 7),
  .splitrule = "variance",
  .min.node.size = 5
)

# Fit random forest: model
model <- train(
  quality~ .,
  tuneGrid = tuneGrid,
  data = wine, 
  method = "ranger",
  trControl = trainControl(
    method = "cv", 
    number = 5, 
    verboseIter = TRUE
  )
)

model
plot(model)

#GLMNET (glmnet models place constraints on your coefficients, which helps prevent overfitting.)
#Classification problems are a little more complicated than regression problems because you have to provide a custom summaryFunction to the train() function to use the AUC metric to rank your models. Start by making a custom trainControl, as you did in the previous chapter. 
#Be sure to set classProbs = TRUE, otherwise the twoClassSummary for summaryFunction will break
# Create custom trainControl: myControl
myControl <- trainControl(
  method = "cv", 
  number = 10,
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE
) #Creating a custome trainControl gives you much finer control over how caret searches for models

# Train glmnet with custom trainControl and tuning: model
overfit <- read.csv("overfit.csv")
model <- train(
  y ~ ., #y is the response variable and all other variables are explanatory variables
  overfit, #dataset
  #custom tuneGrid to explore alpha = 0:1 and 20 values of lambda between 0.0001 and 1 per value of alpha
  #lambda values= control the amount of penalization in the model.
  tuneGrid = expand.grid(alpha = 0:1,
                         lambda = seq(0.0001, 1, length = 20)),
  method = "glmnet",
  trControl = myControl)
model
max(model[["results"]][["ROC"]])

#####PRE_PROCESSING DATA
#Dealing with missing values
#Always try everything and decide the best option empirically

#median imputation
load("~/Desktop/machine_learning/BreastCancer.rdata")
ls()
# Apply median imputation: median_model
median_model <- train(
  x = breast_cancer_x, #x is an object with samples in rows and features in columns (cell size, thickness etc)
  y = breast_cancer_y, #y is a numeric or factor vector containing the outcomes (benign or malignant)
  method = "glm",
  trControl = myControl,
  preProcess = "medianImpute")
median_model

#KNN imputation if data missing is not at random
# Apply KNN imputation: knn_model
knn_model <- train(
  x = breast_cancer_x, 
  y = breast_cancer_y,
  method = "glm",
  trControl = myControl,
  preProcess = "knnImpute")

knn_model
#Compare KNN and median imputation
#resamples() from caret package
resamples <- resamples(x = list(median_model = median_model, knn_model = knn_model))
dotplot(resamples, metric = "ROC") #slightly better is knn model

#####Combining preprocessing methods (always do!!)
# Update model with standardization
model <- train(
  x = breast_cancer_x, 
  y = breast_cancer_y,
  method = "glm",
  trControl = myControl,
  preProcess = c("medianImpute", "center", "scale") #first impute then center and then scale
)

# Print updated model
model

#Remove near zero variance predictors
load("~/Desktop/machine_learning/BloodBrain.rdata")
ls()
#contains many variables and many of these variables have extemely low variances
#caret contains a utility function called nearZeroVar() for removing such variables to save time during modeling
#Identify the near zero variance predictors by running nearZeroVar()
remove_cols <- nearZeroVar(bloodbrain_x, names = TRUE, 
                           freqCut = 2, uniqueCut = 20)

# Get all column names from bloodbrain_x: all_cols
all_cols <- names(bloodbrain_x)

#Make a new data frame called bloodbrain_x_small with the near-zero variance variables removed. Use setdiff() to isolate the column names that you wish to keep (i.e. that you don't want to remove.)
bloodbrain_x_small <- bloodbrain_x[ , setdiff(all_cols, remove_cols)]
model <- train(
  x = bloodbrain_x_small, 
  y = bloodbrain_y, 
  method = "glm"
)
model

#second method is to ruin PCA 
#PCA is generally a better method for handling low-information predictors than throwing them out entirely.
model <- train(
  x = bloodbrain_x, 
  y = bloodbrain_y,
  method = "glm", 
  preProcess = "pca")
model

#############FINAL ANALYSIS
library(C50)
data(churn) #Customer churn is the percentage of customers that stopped using your company's product or service during a certain time frame
#the modeling challenge is to predict which customers will cancel their service (or churn)
ls()
table(churnTrain$churn) / nrow(churnTrain) 
str(churnTrain)
churn_x <- churnTrain[ , -20]
churn_x
churn_y <- churnTrain[ , 20]
churn_y

#Use createFolds() to create 5 CV folds on churn_y, your target variable for this exercise.
myFolds <- createFolds(churn_y, k = 5)

# Create reusable trainControl object: myControl
#By saving the indexes in the train control, we can fit many models using the same CV folds
myControl <- trainControl(
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE,
  savePredictions = TRUE,
  index = myFolds)

#Now that you have a reusable trainControl object called myControl, you can start fitting different predictive models to your churn dataset and evaluate their predictive accuracy.
# Fit glmnet model: model_glmnet
#This model uses our custome CV folds and will be easily compared to other models.
library(caret)
model_glmnet <- train(
  x = churn_x, 
  y = churn_y,
  metric = "ROC",
  method = "glmnet",
  trControl = myControl)
model_glmnet
model_rf <- train(
  x = churn_x, 
  y = churn_y,
  metric = "ROC",
  method = "ranger",
  trControl = myControl)

#comapre these two
# Create model_list
model_list <- list(item1 = model_glmnet, item2 = model_rf)

# Pass model_list to resamples(): resamples
resamples <- resamples(model_list)

# Summarize the results
summary(resamples)

#Create a box-and-whisker plot tocompare
library(caret)
bwplot(resamples, metric = "ROC") #you want the model with the higher median AUC, as well as a smaller range between min and max AUC.

# Create xyplot
xyplot(resamples, metric = "ROC")


save.image("~/Desktop/machine_learning/Machine learning toolbox env.RData")
