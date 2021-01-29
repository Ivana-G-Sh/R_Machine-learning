setwd("~/Desktop/machine_learning")
ata("iris")

#K_MEANS

# Summarize the iris data set
summary(iris)

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

#UNSUPERVISED CLUSTERING (K_MEANS)
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

#performance measure
seeds <- read.csv("seeds.csv")
head(seeds)
# Set random seed. Don't remove this line
set.seed(1)
str(seeds)
# Group the seeds in three clusters
km_seeds <- kmeans(seeds, 3)
# Color the points in the plot based on the clusters
plot(length ~ compactness, data = seeds, col = km_seeds$cluster)
# Print out the ratio of the WSS (within sum of squares) to the BSS ( between cluster sum of squares)
km_seeds$tot.withinss/km_seeds$betweenss 
# The within sum of squares is far lower than the between sum of squares. Indicating the clusters are well seperated and overall compact. 

#k-means cont
set.seed(100)
#Group the seeds in three clusters using kmeans(). 
#Set nstart to 20 to let R randomly select the centroids 20 times. 
str(seeds)
head(seeds)
seeds_km <- kmeans(seeds, 3, nstart=20)
seeds_km
# seeds and seeds_type are pre-loaded in your workspace
seeds_type
str(seeds)
# Do k-means clustering with three clusters, repeat 20 times: seeds_km
seeds_km <- kmeans(seeds, 3, nstart=20) #nstart > 10 uvijek
seeds_km
# Compare clusters with actual seed types (object sed_types). Set k-means clusters as rows
seeds_type <-  c(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3)
table(seeds_km$cluster, seeds_type)
# Plot the length as function of width. Color by cluster
plot(y=seeds$length, x= seeds$width, col=seeds_km$cluster)

#What is your optimal k?
school_result <- read.csv("school_result.csv")
set.seed(100)
str(school_result)
# Initialise ratio_ss 
ratio_ss <- rep(0, 7)

# Finish the for-loop. 
for (k in 1:7) {
  
  # Apply k-means to school_result: school_km
  school_km <- kmeans(school_result, k, nstart=20)
  
  # Save the ratio between of WSS to TSS in kth element of ratio_ss
  ratio_ss[k] <- school_km$tot.withinss/school_km$totss
  
}

# Make a scree plot with type "b" and xlab "k"
plot(ratio_ss, type="b", xlab="k") #with argument b you are connecting the dots
#the optimal number of clusters should at the elbow i.e. 4 or 4

#Standardisation
library(clValid)
run_record <- read.csv("run_record_sc.csv")
head(run_record)
set.seed(1)
# Standardize run_record, transform to a dataframe: run_record_sc
a <- scale(run_record)
run_record_sc <- as.data.frame(a)
# Cluster run_record_sc using k-means: run_km_sc. 5 groups, let R start over 20 times
run_km_sc <- kmeans(run_record_sc, 5, nstart=20)
# Plot records on 100m as function of the marathon. Color using the clusters in run_km_sc
plot(x=run_record_sc$marathon, y=run_record_sc$X100m, col=run_km_sc$cluster)
# Compare the resulting clusters in a nice table
table(run_km_sc$cluster)
# Calculate Dunn's index: dunn_km_sc. Print it.
dunn_km_sc <- dunn(clusters = run_km_sc$cluster, Data = run_record_sc)
dunn_km_sc 

#HIERARCHICAL
library(stats)
#Single clustering
run_record_sc <- read.csv("run_record_sc.csv")
str(run_record_sc)
is.na(run_record_sc)
run_record<-na.omit(run_record_sc)
#Calculate the Euclidean distance matrix of run_record_sc using dist(). Assign it to run_dist. dist() uses the Euclidean method by default.
run_dist <- dist(run_record)
#Use the run_dist matrix to cluster your data hierarchically, based on single-linkage. Use hclust() with two arguments. Assign it to run_single.
run_single <- hclust(run_dist, "single")
#Cut the tree using cutree() at 5 clusters. Assign the result to memb_single.
memb_single <- cutree(run_single, 5)
#Make a dendrogram of run_single using plot(). If you pass a hierarchical clustering object to plot(), it will draw the dendrogram of this clustering.
plot(run_single)
#Draw boxes around the 5 clusters using rect.hclust(). Set the border argument to 2:6, for different colors.
rect.hclust(run_single, k=5, border=2:6)


#Complete Hierarchical Clustering-better
# Apply hclust() to run_dist: run_complete
run_complete <- hclust(run_dist, method = "complete")
head(run_complete)
# Apply cutree() to run_complete: memb_complete
memb_complete <- cutree(run_complete, 5)
# Apply plot() on run_complete to draw the dendrogram
plot(run_complete)
# Apply rect.hclust() on run_complete to draw the boxes
rect.hclust(run_complete, k=5, border = 2:6)
# Compare the membership between the single and the complete linkage clusterings, using table().
table(memb_single, memb_complete)

#Hierarchical vs k-means (Dunns index)
# Set random seed. Don't remove this line.
set.seed(100)
# Dunn's index for k-means: dunn_km
dunn_km <- dunn(clusters = run_km_sc$cluster, Data = run_record_sc)
# Dunn's index for single-linkage: dunn_single
dunn_single <- dunn(clusters = memb_single, Data=run_record_sc)
# Dunn's index for complete-linkage: dunn_complete
dunn_complete <- dunn(clusters = memb_complete, Data=run_record_sc)
# Compare k-means with single-linkage
table(run_km_sc$cluster,memb_single)
# Compare k-means with complete-linkage
table(run_km_sc$cluster,memb_complete)
> dunn_km
[1] 0.1453556
> dunn_single
[1] 0.2921946 #ispada da je obva metoda najbolja no zapravo bas i nije
> dunn_complete
[1] 0.1808437

Clustering US states based on criminal activity