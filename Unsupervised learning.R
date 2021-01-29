###K-means clustering
library(ggplot2)
library(dplyr)
dir()
x <- read.csv("x.csv")
ggplot(x, aes(x=X..1., y=X..2.)) + geom_point()

# Create the k-means model: km.out
km.out <- kmeans(x, centers=3, nstart=20) #start 20 times
# Inspect the result
summary(km.out)
# Print the cluster membership component of the model
print(km.out$cluster)
# Print the km.out object
print(km.out)
## Scatter plot of x - Color the dots on the scatterplot by setting the col argument to the cluster component in km.out
plot(x, col = km.out$cluster, main="k-means with 3 clusters", xlab="",ylab="")

#Handling random algorithms
#always before running iteratively k-means run set seed to ensure reproducibility
# Set up 2 x 3 plotting grid
par(mfrow = c(2, 3))
# Set seed
set.seed(1)
#For each iteration of the for loop, run kmeans() on x. 
#Assume the number of clusters is 3 and number of starts (nstart) is 1
for(i in 1:6) {
  # Run kmeans() on x with three clusters and one start
  km.out <- kmeans(x, centers=3, nstart=1)
  # Plot clusters
  plot(x, col = km.out$cluster, main = km.out$tot.withinss, xlab = "", ylab = "")
} #Because of the random initialization of the k-means algorithm, there's quite some variation in cluster assignments among the six models

#Selecting number of clusters
dev.off() #ponistim formatiranje 2x3
# Initialize total within sum of squares error: wss
wss <- 0
#Build 15 kmeans() models on x, each with a different number of clusters (ranging from 1 to 15). 
#Set nstart = 20 for all model runs and save the total within cluster sum of squares for each model to the ith element of wss.
#Run the code provided to create a scree plot of the wss for all 15 model
# For 1 to 15 cluster centers
for (i in 1:15) {
  km.out <- kmeans(x, centers = i, nstart = 20)
  # Save total within sum of squares to wss variable
  wss[i] <- km.out$tot.withinss}
# Plot total within sum of squares vs. number of clusters
plot(1:15, wss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")

# Set k equal to the number of clusters corresponding to the elbow location
k <- 2 #Looking at the scree plot, it looks like there are inherently 2 or 3 clusters in the data



#Pokemon data
dir()
pokemon <- read.csv("pokemon2.csv")
# Initialize total within sum of squares error: wss
wss <- 0
# Look over 1 to 15 possible clusters
for (i in 1:15) {
  # Fit the model: km.out
  km.out <- kmeans(pokemon, centers = i, nstart = 20, iter.max = 50) #default number of iterations for kmeans is 10
  # Save the within cluster sum of squares
  wss[i] <- km.out$tot.withinss}

# Produce a scree plot
plot(1:15, wss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")

# Select number of clusters
k <- 2
# Build model with k clusters: km.out
km.out <- kmeans(pokemon, centers = 2, nstart = 20, iter.max = 50)
# View the resulting model
km.out
# Plot of Defense vs. Speed by cluster membership
plot(pokemon[, c("Defense", "Speed")], col = km.out$cluster,
     main = paste("k-means clustering of Pokemon with", k, "clusters"),
     xlab = "Defense", ylab = "Speed")
save.image("~/Desktop/machine_learning/Unsupervised learning/env.RData")

#hijerarhical clustering
#on the basis of standard euclidian distance
# Create hierarchical clustering model: hclust.out
dir()
x <- read.csv("x_hierarchical.csv")
hclust.out <- hclust(dist(x))

# Inspect the result
summary(hclust.out) #not so useful and clear

#selecting number of clusters - cutting the tree i.e. dendogram
#h and k arguments to cutree() allow you to cut the tree based on a certain height h or a certain number of clusters k

# Cut by height (kao da od y osi ide ravna lionija na visini 7)
cutree(hclust.out, h=7)
# Cut by number of clusters (arbitrarno)
cutree(hclust.out, k=3) #The output of each cutree() call represents the cluster assignments for each observation in the original dataset. Great work!

#linkage models (how do you link the clusters together)
# Cluster using complete linkage: hclust.complete
hclust.complete <- hclust(dist(x), method = "complete")
# Cluster using average linkage: hclust.average
hclust.average <- hclust(dist(x), method = "average")
# Cluster using single linkage: hclust.single
hclust.single <- hclust(dist(x), method = "single")
# Plot dendrogram of hclust.complete
plot(hclust.complete, main="Complete")

# Plot dendrogram of hclust.average
plot(hclust.average, main = "Average")

# Plot dendrogram of hclust.single
plot(hclust.single, main = "Single")

#najizbalansiraniji su complete i average
# Whether you want balanced or unbalanced trees for your hierarchical clustering model depends on the context of the problem you're trying to solve. 
#Balanced trees are essential if you want an even number of observations assigned to each cluster. On the other hand, if you want to detect outliers, for example, an unbalanced tree is more desirable because pruning an unbalanced tree can result in most observations assigned to one cluster and only a few observations assigned to other clusters.

#Practical matters: scaling
#Observe the mean of each variable in pokemon using the colMeans() function.
# View column means
colMeans(pokemon)
#Observe the standard deviation of each variable using the apply() and sd() functions. 
#Since the variables are the columns of your matrix, make sure to specify 2 as the MARGIN argument to apply().
# View column standard deviations
apply(pokemon, 2, sd)
#Scale the pokemon data using the scale() function and store the result in pokemon.scaled.
# Scale the data
pokemon.scaled <- scale(pokemon)

#Create a hierarchical clustering model of the pokemon.scaled data using the complete linkage method.
# Manually specify the method argument and store the result in hclust.pokemon.
hclust.pokemon <- hclust(dist(pokemon.scaled), method = "complete" )

#Comparing kmeans() and hclust()
#Comparing k-means and hierarchical clustering, you'll see the two methods produce different cluster memberships

#the results from running k-means clustering on the pokemon data (for 3 clusters) are stored as km.pokemon. The hierarchical clustering model you created in the previous exercise is still available as hclust.pokemon
# Apply cutree() to hclust.pokemon: cut.pokemon
cut.pokemon <- cutree(hclust.pokemon, k=3)

# Compare methods
table(cut.pokemon, km.pokemon$cluster) #nema consensusa koja je metoda bolja. treba obje probati

#PCA plots (dimensionality reduction)
#Create a PCA model of the data in pokemon, setting scale to TRUE. Store the result in pr.out.
pr.out <- prcomp(x=pokemon, scale=TRUE, center=TRUE)
#Inspect the result with the summary() function.
summary(pr.out)
#What is the minimum number of principal components that are required to describe at least 75% of the cumulative variance in this dataset? 
# the first tri components describe 76% of the variance in the dataset
#BIPLOT
biplot(pr.out)
#SCREE PLOT
#Assign to the variable pr.var the square of the standard deviations of the principal components (i.e. the variance). The standard deviation of the principal components is available in the sdev component of the PCA model object.
pr.var <- pr.out$sdev^2
#Assign to the variable pve the proportion of the variance explained, calculated by dividing pr.var by the total variance explained by all principal components.
pve <- pr.var / sum(pr.var)
#One way to determine the number of principal components to retain is by looking for an elbow in the scree plot showing that as the number of principal components increases, the rate at which variance is explained decreases substantially. In the absence of a clear elbow, you can use the scree plot as a guide for setting a threshold.
# Plot variance explained for each principal component
plot(pve, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     ylim = c(0, 1), type = "b")

# Plot cumulative proportion of variance explained
plot(cumsum(pve), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     ylim = c(0, 1), type = "b")
#when the number of principal components is equal to the number of original features in the data, the cumulative proportion of variance explained is 1

#SCALING
#Run this code to see how the scale of the variables differs in the original data.
# Mean of each variable
colMeans(pokemon)

# Standard deviation of each variable
apply(pokemon, 2, sd)
# PCA model with scaling: pr.with.scaling
pr.with.scaling <- prcomp(x=pokemon, scale=TRUE, center=TRUE)

# PCA model without scaling: pr.without.scaling
pr.without.scaling <- prcomp(x=pokemon, scale=FALSE, center=TRUE)

# Create biplots of both for comparison
biplot(pr.with.scaling)
biplot(pr.without.scaling)


#######################CASE ANALYSIS#######################
url <- "http://s3.amazonaws.com/assets.datacamp.com/production/course_1903/datasets/WisconsinCancer.csv"

#Use read.csv() function to download the CSV (comma-separated values) file containing the data from the URL provided. Assign the result to wisc.df.
# Download the data: wisc.df
wisc.df <- read.csv(url)
head(wisc.df)
#Use as.matrix() to convert the features of the data (in columns 3 through 32) to a matrix. Store this in a variable called wisc.data.
wisc.data <- as.matrix(wisc.df[3:32])
head(wisc.data)
#Assign the row names of wisc.data the values currently contained in the id column of wisc.df. While not strictly required, this will help you keep track of the different observations throughout the modeling process.
row.names(wisc.data) <- wisc.df$id
#Finally, set a vector called diagnosis to be 1 if a diagnosis is malignant ("M") and 0 otherwise. Note that R coerces TRUE to 1 and FALSE to 0.
diagnosis <- as.numeric(wisc.df$diagnosis == "M")

#How many observations are in this dataset?569, 10, 212
glimpse(wisc.data)
#How many variables/features in the data are suffixed with _mean?
head(wisc.data)
#How many of the observations have a malignant diagnosis?
wisc.data[diagnosis]=="M" ??

# Check the mean and standard deviation of the features of the data to determine if the data should be scaled. Use the colMeans() and apply() functions like you've done before.
colMeans(wisc.data)
apply(wisc.data, 2, sd)

# Execute PCA, scaling if appropriate: wisc.pr
wisc.pr <- prcomp(x=wisc.data, scale=TRUE, center=TRUE)

# Look at summary of results
summary(wisc.pr)

#Create a biplot of the wisc.pr data. What stands out to you about this plot? Is it easy or difficult to understand? Why?
biplot(wisc.pr)

## Scatter plot observations by components 1 and 2
plot(wisc.pr$x [, c(1, 2)], col = (diagnosis + 1), 
     xlab = "PC1", ylab = "PC2")
# Scatter plot observations by components 1 and 2
plot(wisc.pr$x[, c(1, 2)], col = (diagnosis + 1), 
     xlab = "PC1", ylab = "PC2")

# Repeat for components 1 and 3
plot(wisc.pr$x[, c(1, 3)], col = (diagnosis + 1), 
     xlab = "PC1", ylab = "PC3") #Because principal component 2 explains more variance in the original data than principal component 3, you can see that the first plot has a cleaner cut separating the two subgroups.


#In this exercise, you will produce scree plots showing the proportion of variance explained as the number of principal components increases.

#Calculate the variance of each principal component by squaring the sdev component of wisc.pr. Save the result as an object called pr.var
# Set up 1 x 2 plotting grid
par(mfrow = c(1, 2))

# Calculate variability of each component
pr.var <- wisc.pr$sdev^2

#Calculate the variance explained by each principal component by dividing by the total variance explained of all principal components. Assign this to a variable called pve
pve <- pr.var / sum(pr.var)

# Plot variance explained for each principal component
plot(pve, xlab = "Principal Component", 
     ylab = "Proportion of Variance Explained", 
     ylim = c(0, 1), type = "b")

# Plot cumulative proportion of variance explained
plot(cumsum(pve), xlab = "Principal Component", 
     ylab = "Cumulative Proportion of Variance Explained", 
     ylim = c(0, 1), type = "b")

#What is the minimum number of principal components needed to explain 80% of the variance in the data? Write it down as you may need this in the next exercise :) oko 4,5

#For the first principal component, what is the component of the loading vector for the feature concave.points_mean? 
wisc.pr$x
-0.26
#What is the minimum number of principal components required to explain 80% of the variance of the data? Oko 4,5
summary(wisc.pr) 

# Scale the wisc.data data: data.scaled
data.scaled <- scale(wisc.data)

# Calculate the (Euclidean) distances: data.dist
data.dist <- dist(data.scaled)

# Create a hierarchical clustering model: wisc.hclust
wisc.hclust <- hclust(data.dist, method="complete")

#sing the plot() function, what is the height at which the clustering model has 4 clusters?
plot(wisc.hclust) #20

#Use cutree() to cut the tree so that it has 4 clusters. Assign the output to the variable wisc.hclust.clusters.
wisc.hclust.clusters <- cutree(wisc.hclust, k=4)
#Use the table() function to compare the cluster membership to the actual diagnoses. compare the cluster membership to diagnosis, the vector that contains the actual diagnoses?
table(wisc.hclust.clusters, diagnosis)

#Create a k-means model on wisc.data, assigning the result to wisc.km. 
#Be sure to create 2 clusters, corresponding to the actual number of diagnosis. 
#Also, remember to scale the data and repeat the algorithm 20 times to find a well performing model.
wisc.km <- kmeans(x=scale(wisc.data), centers=2, nstart=20) 

# Compare k-means to actual diagnoses
table(wisc.km$cluster, diagnosis)

# Compare k-means to hierarchical clustering
table(wisc.km$cluster, wisc.hclust.clusters) #Looking at the second table you generated, it looks like clusters 1, 2, and 4 from the hierarchical clustering model can be interpreted as the cluster 1 equivalent from the k-means algorithm, and cluster 3 can be interpreted as the cluster 2 equivalent.

#Clustering on PCA results
#Using the minimum number of principal components required to describe at least 90% of the variability in the data, create a hierarchical clustering model with complete linkage. 
#Assign the results to wisc.pr.hclust
summary(wisc.pr) #do PC7
wisc.pr.hclust <- hclust(dist(wisc.pr$x[, 1:7]), method = complete)

#Cut this hierarchical clustering model into 4 clusters and assign the results to wisc.pr.hclust.clusters
# Cut model into 4 clusters: wisc.pr.hclust.clusters
wisc.pr.hclust.clusters <- cutree(wisc.pr.hclust, k=4)

#Using table(), compare the results from your new hierarchical clustering model with the actual diagnoses. How well does the newly created model with four clusters separate out the two diagnoses?
table(wisc.pr.hclust.clusters, diagnosis)

#How well do the k-means and hierarchical clustering models you created in previous exercises do in terms of separating the diagnoses? 
table(wisc.km$cluster, diagnosis)
table(wisc.hclust.clusters, diagnosis)
