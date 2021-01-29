setwd("~/Desktop/machine_learning")

library("ISLR")
library(dplyr)
data(Wage)
class(Wage)
head(Wage)
View(Wage)

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

#quality of regression
air <- read.csv("air.csv", header=TRUE)
head(air)
# Inspect your colleague's code to build the model
fit <- lm(dec ~ freq + angle + ch_length, data = air)
# Use the model to predict for all values: pred
# Use the model to predict for all values: pred
pred <- predict(fit)
# Use air$dec and pred to calculate the RMSE 
rmse <- sqrt((1/nrow(air)) * sum((air$dec - pred)^ 2))
# Print out rmse
rmse #5 decibels but what does it mean, compare with another  model
# Your colleague's more complex model
fit2 <- lm(dec ~ freq + angle + ch_length + velocity + thickness, data = air)
# Use the model to predict for all values: pred2
pred2 <- predict(fit2)
# Calculate rmse2
rmse2 <- sqrt(sum( (air$dec - pred2) ^ 2) / nrow(air))
# Print out rmse2
rmse2

#Simple linear regression
kang_nose <-read.csv("kang_nose.csv")
class(kang_nose)
# Plot nose length as function of nose width.
plot(kang_nose, xlab = "nose width", ylab = "nose length")
# describe the linear relationship between the two variables: lm_kang
lm_kang <- lm(nose_length ~ nose_width, data = kang_nose)
plot(kang_nose, xlab = "nose width", ylab = "nose length")
abline(lm_kang$coefficients, col = "red")
# Print the coefficients of lm_kang
lm_kang$coefficients
#predict the nose witdt of a new kangaroo on the basisi of its nose lenght
nose_width_new <- read.csv("nose_width_new.csv")
nose_width_new
# Predict and print the nose length of the escaped kangoroo
nose_width_new
predict(lm_kang, nose_width_new)
#RMSE:
# Apply predict() to lm_kang: nose_length_est
nose_length_est <- predict(lm_kang)
# Calculate difference between the predicted and the true values: res
res <- kang_nose$nose_length-nose_length_est
# Calculate RMSE, assign it to rmse and print it
rmse <- sqrt(mean (res ^ 2) )
rmse #the value is 43 mm but what does it mean?
#treba ti R^2 - to moze izracunati sam ili pomocu summary
# Calculate the residual sum of squares: ss_res
ss_res=sum(res^2)
# Determine the total sum of squares: ss_tot
average_nose_lenght<- mean(kang_nose$nose_length)
ss_tot=sum((kang_nose$nose_length-average_nose_lenght)^2)
# Calculate R-squared and assign it to r_sq. Also print it.
r_sq= 1-ss_res / ss_tot
r_sq
# Apply summary() to lm_kang
summary(lm_kang)$r.squared #resultst od R-squared of 0.77 is pretty neat!

#log-linear model
world_bank_train <- read.csv("world_bank_train.csv.csv")
head(world_bank_train)
plot(urb_pop ~ cgdp, data = world_bank_train,
     xlab = "GDP per Capita",
     ylab = "Percentage of urban population")
#convert cgdp to log - it will look better
plot(urb_pop ~ log(cgdp), data = world_bank_train,
     xlab = "log(GDP per Capita)",
     ylab = "Percentage of urban population")
#linear model
lm_wb <- lm(urb_pop ~ log(cgdp), data = world_bank_train)
# Linear model: change the formula
lm_wb <- lm(urb_pop ~ log(cgdp), data = world_bank_train)
# Add a red regression line to your scatter plot
plot(urb_pop ~ log(cgdp), data = world_bank_train,
     xlab = "log(GDP per Capita)",
     ylab = "Percentage of urban population")
abline(lm_wb$coefficients, col = "red")
# Summarize lm_wb and select R-squared
summary(lm_wb)$r.squared
#predict %of urban pop in afganistan on the basiis of cgdp
cgdp_afg<- read.csv("afg.csv")
predict(lm_wb, cgdp_afg)

#RMSE
shop_data <- read.csv("shop_data.csv")
# Add a plots. Is linearity plausible?
plot(sales ~ sq_ft, shop_data)
plot(sales ~ size_dist, shop_data)
plot(sales ~ inv, shop_data)
# Build a linear model for net sales based on all other variables: lm_shop. Use the formula sales ~ . to include all variables
lm_shop <- lm(sales ~ ., data=shop_data)
# Summarize lm_shop
summary(lm_shop)
summary(lm_shop)$r.squared
summary(lm_shop)$adj.r.squared
#Are all predictors relevant?
# Plot the residuals in function of your fitted observations
plot(x=lm_shop$fitted.values, y=lm_shop$residuals) #there should be no clear pattern
# Make a Q-Q plot of your residual quantiles
qqnorm(lm_shop$residuals, ylab="Residual Quantiles") #should be a straigt line
# Summarize your model, are there any irrelevant predictors?
summary(lm_shop) #From the small p-values you can conclude that every predictor is important!
# Predict the net sales based on shop_new
shop_new <- read.csv("shop_new.csv")
predict(lm_shop, shop_new)

#non-linear models (training dataste and test)
world_bank_train <- read.csv("world_bank_train.csv.csv")
lm_wb_log <- lm(urb_pop ~ log(cgdp), data = world_bank_train)
# Calculate rmse_train
rmse_train <- sqrt(mean(lm_wb_log$residuals ^ 2))
# The real percentage of urban population in the test set, the ground truth
world_bank_test <- read.csv("world_bank_test.csv")
world_bank_test_truth <- world_bank_test$urb_pop
# The predictions of the percentage of urban population in the test set
world_bank_test_input <- data.frame(cgdp = world_bank_test$cgdp)
world_bank_test_output <- predict(lm_wb_log, world_bank_test_input)
# The residuals: the difference between the ground truth and the predictions
res_test <- world_bank_test_output - world_bank_test_truth
# Use res_test to calculate rmse_test
rmse_test <- sqrt(mean (res_test ^ 2) )
# Print the ratio of the test RMSE over the training RMSE
rmse_test/rmse_train #The test's RMSE is only slightly larger than the training RMSE. This means that your model generalizes well to unseen observations.



#multivariate linear model -K_NN
world_bank_train <- read.csv("world_bank_train.csv.csv")
world_bank_test <- read.csv("world_bank_test.csv")
# inspect it and try to understand how it works!
my_knn <- function(x_pred, x, y, k){
  m <- length(x_pred)
  predict_knn <- rep(0, m)
  for (i in 1:m) {
    
    # Calculate the absolute distance between x_pred[i] and x
    dist <- abs(x_pred[i] - x)
    
    # Apply order() to dist, sort_index will contain
    # the indices of elements in the dist vector, in
    # ascending order. This means sort_index[1:k] will
    # return the indices of the k-nearest neighbors.
    sort_index <- order(dist)
    
    # Apply mean() to the responses of the k-nearest neighbors
    predict_knn[i] <- mean(y[sort_index[1:k]])
    
  }
  return(predict_knn)
}
###

# Apply your algorithm on the test set: test_output
test_output <- my_knn( world_bank_test$cgd, world_bank_train$cgdp, world_bank_train$urb_pop, k=30)
# Have a look at the plot of the output
plot(world_bank_train,
     xlab = "GDP per Capita",
     ylab = "Percentage Urban Population")
points(world_bank_test$cgdp, test_output, col = "green")



##Which model performs the best?
# Define ranks to order the predictor variables in the test set
ranks <- order(world_bank_test$cgdp)

# Predict with simple linear model and add line
test_output_lm <- predict(lm_wb, data.frame(cgdp = world_bank_test$cgdp))
plot(world_bank_test,
     xlab = "GDP per Capita", ylab = "Percentage Urban Population")

# Predict with log-linear model and add line
test_output_lm_log <- predict(lm_wb_log, data.frame(cgdp = world_bank_test$cgdp))

# Predict with k-NN and add line
test_output_knn <- my_knn( world_bank_test$cgd, world_bank_train$cgdp, world_bank_train$urb_pop, 30)

plot(world_bank_test,
     xlab = "GDP per Capita", ylab = "Percentage Urban Population")
lines(world_bank_test$cgdp[ranks], test_output_lm[ranks], lwd = 2, col = "blue")
lines(world_bank_test$cgdp[ranks], test_output_lm_log[ranks], lwd = 2, col = "red")
lines(world_bank_test$cgdp[ranks], test_output_knn[ranks], lwd = 2, col = "green")

# Calculate RMSE on the test set for simple linear model
sqrt(mean((test_output_lm - world_bank_test$urb_pop) ^ 2))

# Calculate RMSE on the test set for log-linear model
sqrt(mean((test_output_lm_log - world_bank_test$urb_pop) ^ 2))

# Calculate RMSE on the test set for k-NN technique
sqrt(mean((test_output_knn - world_bank_test$urb_pop) ^ 2))




