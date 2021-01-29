setwd("~/Desktop/machine_learning")
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)


#1.classificiation with decision tree
titanic <- read.csv("titanic.csv", header = TRUE)
head(titanic)

# In problems that have a random aspect, the set.seed() function is used to enforce reproducibility
set.seed(1)

# Shuffle the dataset, call the result shuffled
n <- nrow(titanic)
shuffled <- titanic[sample(n),]

#Split the dataset into a train set, and a test set. Use a 70/30 split. 
train_indices <- 1:round(0.7 * n)
train <- shuffled[train_indices, ]
test_indices <- (round(0.7 * n) + 1):n
test <- shuffled[test_indices, ]
# Print the structures
str(train)
str(test)
# This is a decision tree - Survived is  a function that I am trying to learn. DOt is instead of Pclass+sex+age. TRain is the training dataset. Method+class denotes it si a classification probelm
tree <- rpart(Survived ~ ., train, method = "class")
  ##in the previous line the default splitting criterion is gini. You can change it to some other and compare the accuracy of both
  ## Change the first line of code to use information gain as splitting criterion
  tree_i <- rpart(spam ~ ., train, method = "class", parms = list(split = "information"))
# Draw the decision tree
??fancyRpartplot
fancyRpartPlot(tree)
#Use the predict() function with the tree model as the first argument and the correct dataset as the second argument. Set type to "class". Call the predicted vector pred. Remember that you should do the predictions on the test set.
pred <- predict(tree, test, type="class")
#Use the table() function to calculate the confusion matrix. Assign this table to conf. Construct the table with the test set's actual values (test$Survived) as the rows and the test set's model predicted values (pred) as columns.
conf <- table(test$Survived, pred)
# Print this confusion matrix
conf
## Assign TP, FN, FP and TN using conf
TP <- conf[1, 1] # true positive
FN <- conf[1, 2] # false negative
FP <- conf[2, 1] # false positive
TN <- conf[2, 2] # true nehative

# Calculate and print the accuracy: acc
acc <- (TP+TN)/(TP+FN+FP+TN)
acc <- sum(diag(conf))/sum(conf)
acc
# Calculate and print out the precision: prec
prec <- TP/(TP+FP)
prec
# Calculate and print out the recall: rec
rec <- TP/(TP+FN)
rec

#Prunning

# Calculation of a complex tree
set.seed(1)
tree <- rpart(Survived ~ ., train, method = "class", control = rpart.control(cp=0.00001))
# Draw the complex tree
fancyRpartPlot(tree)
# Prune the tree: pruned
pruned<- prune(tree,cp=0.01)
# Draw pruned
fancyRpartPlot(pruned)


#using cross-validation (build a model n times and compare conf matrix)
set.seed(1)

# The shuffled dataset is already loaded into your workspace

# Initialize the accs vector
accs <- rep(0,6)

for (i in 1:6) {
  # These indices indicate the interval of the test set
  indices <- (((i-1) * round((1/6)*nrow(shuffled))) + 1):((i*round((1/6) * nrow(shuffled))))
  # Exclude them from the train set
  train <- shuffled[-indices,]
  # Include them in the test set
  test <- shuffled[indices,]
  # A model is learned using each training set
  tree <- rpart(Survived ~ ., train, method = "class")
  # Make a prediction on the test set using tree
  pred <- predict(tree, test, type = "class")
  # Assign the confusion matrix to conf
  conf <- table(test$Survived, pred)
  # Assign the accuracy of this model to the ith index in accs
  accs[i] <- sum(diag(conf))/sum(conf)
}

# Print out the mean of accs
mean(accs)

#Fitting:

#Classification
emails <- read.csv("emails_small.csv")
head(emails)
dim(emails)
#So where the avg_capital_seq is greater than 4, spam_classifier() predicts the email is spam (1), if avg_capital_seq is inclusively between 3 and 4, it predicts not spam (0), and so on. 
# Inspect definition of spam_classifier()
spam_classifier <- function(x){
  prediction <- rep(NA, length(x)) # initialize prediction vector
  prediction[x > 4] <- 1
  prediction[x >= 3 & x <= 4] <- 0
  prediction[x >= 2.2 & x < 3] <- 1
  prediction[x >= 1.4 & x < 2.2] <- 0
  prediction[x > 1.25 & x < 1.4] <- 1
  prediction[x <= 1.25] <- 0
  return(prediction) # prediction is either 0 or 1
}

# Apply the classifier to the avg_capital_seq column: spam_pred
spam_pred <- spam_classifier(emails$avg_capital_seq)
spam_pred

# Compare spam_pred to emails$spam. Use ==
spam_pred == emails$spam #accuracy is 100%

emails_full <- read.csv("emails_full.csv")
View(emails_full)
spam_classifier <- function(x){
  prediction <- rep(NA, length(x)) # initialize prediction vector
  prediction[x > 4] <- 1 
  prediction[x >= 3 & x <= 4] <- 0
  prediction[x >= 2.2 & x < 3] <- 1
  prediction[x >= 1.4 & x < 2.2] <- 0
  prediction[x > 1.25 & x < 1.4] <- 1
  prediction[x <= 1.25] <- 0
  return(factor(prediction, levels = c("1", "0"))) # prediction is either 0 or 1
}

# Apply spam_classifier to emails_full: pred_full
pred_full <- spam_classifier(emails_full$avg_capital_seq)

# Build confusion matrix for emails_full: conf_full
conf_full <- table(emails_full$spam, pred_full)

# Calculate the accuracy with conf_full: acc_full
acc_full <- sum(diag(conf_full))/sum(conf_full)

# Print acc_full
acc_full
#This hard-coded classifier gave you an accuracy of around 65% on the full dataset, which is way worse than the 100% you had on the small dataset back in chapter 1. Hence, the classifier does not generalize well at all!

####################################################
# 2. classification with k-Nearest Neighbors (k-NN)
library(class)
# train and test are pre-loaded
str(train)
str(test)
# Store the Survived column of train and test in train_labels and test_labels
train_labels <- train$Survived
test_labels <- test$Survived
# Copy train and test to knn_train and knn_test
knn_train <- train
knn_test <- test
# Drop Survived column for knn_train and knn_test
knn_train$Survived <-NULL
knn_test$Survived<-NULL
# Normalize Pclass
min_class <- min(knn_train$Pclass)
max_class <- max(knn_train$Pclass)
knn_train$Pclass <- (knn_train$Pclass - min_class) / (max_class - min_class)
knn_test$Pclass <- (knn_test$Pclass - min_class) / (max_class - min_class)

# Normalize Age
min_age <- min(knn_train$Age)
max_age <- max(knn_train$Age)
knn_train$Age <- (knn_train$Age - min_age) / (max_age - min_age)
knn_test$Age <- (knn_test$Age - min_age) / (max_age - min_age)

?knn()
# Set random seed. Don't remove this line.
set.seed(1)
# make predictions using knn: pred
#train: observations in the training set, without the class labels, available in knn_train
#test: observations in the test, without the class labels, available in knn_test
#cl: factor of true class labels of the training set, available in train_labels
#k: number of nearest neighbors you want to consider, 5 in our case
pred <- knn(train = knn_train, test = knn_test, cl = train_labels, k = 5)
# Construct the confusion matrix: conf
conf <-table(test_labels, pred)
conf

#choice of a suitable k
train_labels <- train$Survived
test_labels <- test$Survived
# Copy train and test to knn_train and knn_test
knn_train <- train
knn_test <- test
knn_train$Survived <-NULL
knn_test$Survived<-NULL
min_class <- min(knn_train$Pclass)
max_class <- max(knn_train$Pclass)
knn_train$Pclass <- (knn_train$Pclass - min_class) / (max_class - min_class)
knn_test$Pclass <- (knn_test$Pclass - min_class) / (max_class - min_class)
min_age <- min(knn_train$Age)
max_age <- max(knn_train$Age)
knn_train$Age <- (knn_train$Age - min_age) / (max_age - min_age)
knn_test$Age <- (knn_test$Age - min_age) / (max_age - min_age)
knn_train$Pclass <- (knn_train$Pclass - min_class) / (max_class - min_class)
knn_test$Pclass <- (knn_test$Pclass - min_class) / (max_class - min_class)
set.seed(1)

# Load the class package, define range and accs
library(class)
range <- 1:round(0.2 * nrow(knn_train))
accs <- rep(0, length(range))

for (k in range) {
  pred <- knn(train = knn_train, test = knn_test, cl = train_labels, k = k)
  conf <- table(test_labels, pred)
  accs[k] <- sum(diag(conf))/sum(conf)
}

# Plot the accuracies. Title of x-axis is "k".
plot(range, accs, xlab = "k")
# Calculate the best k
which.max(accs) #trebalo bi biti 73

#ROC curves
#nemnam dataset
library(ROCR)
set.seed(1)
# Build a tree on the training set: tree
tree <- rpart(Survived ~ ., train, method = "class")
# Predict probability values using the model: all_probs
#The first argument should be the tree model that is built, tree
#The second argument should be the test set, on which you want to predict
#Finally, don't forget to set type to "prob".
all_probs <- predict(tree, test, type="prob")
# Print out all_probs
head(all_probs)
class(all_probs) #its a matrix
all_probs
# Select second column of all_probs: probs
probs <- all_probs[,2]
probs
pred <- prediction(probs, test$Survived)
# Make a performance object: perf
perf <- performance(pred, "tpr", "fpr")
# Plot ROC curve
plot(perf)
#get AUC
perf <- performance(pred, "auc")
perf@y.values[[1]] #printed out AUC, 0.85 is a good value

#comparison of decision tree and K_NN
# samo nacrtas ROC curves od oba

