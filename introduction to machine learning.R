setwd("~/Desktop/machine_learning")
data("iris")
# Reveal number of observations and variables in two different ways
dim(iris)
str(iris)

# Show first and last observations in the iris data set
head(iris)
tail(iris)

# Summarize the iris data set
summary(iris)

#Basic prediction model
library("ISLR")
data(Wage)
class(Wage)
head(Wage)
View(Wage)



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
spam_pred == emails$spam

#measuring performance of classification: confusion matrix
titanic <- read.csv("titanic.csv", header = TRUE)
head(titanic)
# Set random seed. Don't remove this line
set.seed(1)
# Have a look at the structure of titanic
str(titanic)
# A decision tree classification model is built on the data
tree <- rpart(Survived ~ ., data = titanic, method = "class")
# Use the predict() method to make predictions, assign to pred
# The tree aims to predict whether a person would have survived the accident based on the variables Age, Sex and Pclass (travel class). The decision the tree makes can be deemed correct or incorrect if we know what the person's true outcome was. That is, if it's a supervised learning problem.
pred <- predict(tree, titanic, type="class")

#Since the true fate of the passengers, Survived, is also provided in titanic, you can compare it to the prediction made by the tree. As you've seen in the video, the results can be summarized in a confusion matrix. In R, you can use the table() function for this.
# Use the table() method to make the confusion matrix
conf <- table(titanic$Survived, pred)
# Assign TP, FN, FP and TN using conf
TP <- conf[1, 1] # true positive
FN <- conf[1, 2] # false negative
FP <- conf[2, 1] # false positive
TN <- conf[2, 2] # true nehative

# Calculate and print the accuracy: acc
acc <- (TP+TN)/(TP+FN+FP+TN)
acc
# Calculate and print out the precision: prec
prec <- TP/(TP+FP)
prec
# Calculate and print out the recall: rec
rec <- TP/(TP+FN)
rec

#Regression 
#wage of a worker based only on the worker's age.
lm_wage <- lm(wage ~ age, data = Wage)
#Define data.frame: unseen (coded already)
unseen <- data.frame(age = 60)
# Predict the wage for a 60-year old worker
predict(lm_wage, unseen)


#LinkedIn views for the next 3 days
linkedin <- c(5,  7,  4,  9, 11, 10, 14, 17, 13, 11, 18, 17, 21, 21, 24, 23, 28, 35, 21, 27, 23)
class(linkedin)

# Create the days vector
days <- c(1:21)

#do a lm (lm(y ~ x) builds a linear model such that y is a function of x)
linkedin_lm <- lm(linkedin ~ days)

# Predict the number of views for the next three days: linkedin_pred
future_days <- data.frame(days = 22:24)
linkedin_pred <- predict(linkedin_lm, future_days)

# Plot historical data and predictions
plot(linkedin ~ days, xlim = c(1, 24))
points(22:24, linkedin_pred, col = "green")

#perforance measure method fro regression is RMSE (root means squared error - mean distances between estimates and regression line)



#Clustering: Separating the iris species
data("iris")
# Set random seed. Don't remove this line.
set.seed(1)

# Chop up iris in my_iris and species
my_iris <- iris[-5]
my_iris
species <- iris$Species
species

# Perform k-means clustering on my_iris: kmeans_iris
kmeans_iris <- kmeans(my_iris, 3)
kmeans_iris
kmeans_iris$cluster

# Compare the actual Species to the clustering using table()
table(species, kmeans_iris$cluster)

# Plot Petal.Width against Petal.Length, coloring by cluster
plot(Petal.Length ~ Petal.Width, data = my_iris, col = kmeans_iris$cluster)

#supervised learning using decision trees
set.seed(1)
str(iris)
summary(iris)

#the model has been built for you
library(rpart)
tree <- rpart(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,data = iris, method = "class")

# A dataframe containing unseen observations
unseen <- data.frame(Sepal.Length = c(5.3, 7.2),
                     Sepal.Width = c(2.9, 3.9),
                     Petal.Length = c(1.7, 5.4),
                     Petal.Width = c(0.8, 2.3))

# Predict the label of the unseen observations. Print out the result.
predict(tree, unseen, type="class")

#UNSUPERVISED CLUSTERING
cars <- read.csv("cars.csv")
set.seed(1)
str(cars)
summary(cars)

# Group the dataset into two clusters: km_cars
km_cars <- kmeans(cars, center=2)
km_cars
# Print out the contents of each cluster
km_cars$cluster

# color the points in the plot based on the clusters 
plot(cars, col = km_cars$cluster)

# Print out the cluster centroids
km_cars$centers

#  add the centroids to the plot
points(km_cars$centers, pch = 22, bg = c(1, 2), cex = 2)

#performance measure: similarity within cluster (within cluster sum of square) and similarity between clusters (intercluser distance)