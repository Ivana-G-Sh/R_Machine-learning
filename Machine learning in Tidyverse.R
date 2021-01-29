#####Machine Learning in the Tidyverse

library(tidyr)
library(purrr)
library(broom)
library(tidyverse)
dir()
gapminder <- readRDS( "gapminder.rds" )

#working with nested dataframes in a tibble
# Explore gapminder
head(gapminder)
# Prepare the nested dataframe gap_nested
gap_nested <- gapminder %>% group_by(country) %>% nest()
# Explore gap_nested
head(gap_nested) #Notice that each row in gap_nested contains a tibble.

#un-nest
# Create the unnested dataframe called gap_unnnested
gap_unnested <- gap_nested %>% unnest()
# Confirm that your data was not modified  
identical(gapminder, gap_unnested) #nest function only reshapes the data and does not modifies it

#Explore a nested cell
# Extract the data of Algeria
algeria_df <- gap_nested$data[[1]]

# Calculate the minimum of the population vector
min(algeria_df$population)
# Calculate the maximum of the population vector
max(algeria_df$population)
# Calculate the mean of the population vector
mean(algeria_df$population) #working with a single chunk in a nested dataframe is identical to working with regular dataframes


#scale this approach to work on a vector of nested dataframes using the map family of functions
# Calculate the mean population for each country
pop_nested <- gap_nested %>%mutate(mean_pop = map(data, ~mean(.x$population)))

# Take a look at pop_nested
head(pop_nested)
# Extract the mean_pop value by using unnest
pop_mean <- pop_nested %>% unnest(mean_pop)


#When you know that the output of your mapped function is an expected type (here it is a numeric vector) you can leverage the map_*() family of functions to explicitly try to return that object type instead of a list.
#Here you will again calculate the mean population of each country, but instead, you will use map_dbl() to explicitly append the numeric vector returned by mean() to your dataframe.
# Calculate mean population and store result as a double
pop_mean <- gap_nested %>% mutate(mean_pop = map_dbl(data, ~mean(.x$population)))
# Take a look at pop_mean
head(pop_mean)

#Mapping many models
# Build a linear model for each country
glimpse(gap_nested)
gap_models <- gap_nested %>%
  mutate(model = map(data, ~lm(formula = life_expectancy~year, data = .x))) #gap_models dataframe contains a model predicting life expectancy by year for 77 countries.

# Extract the model for Algeria    
algeria_model <- gap_models$model[[1]]
# View the summary for the Algeria model
summary(algeria_model)

##Tidy your models with broom
#For a linear model, tidy() extracts the model coefficients while glance() returns the model statistics such as the R2.
library(broom)

# Extract the coefficients of the algeria_model as a dataframe
tidy(algeria_model)

# Extract the coefficient statistics of each model into nested dataframes
model_coef_nested <- gap_models %>% 
  mutate(coef = map(model, ~tidy(.x)))

# Simplify the coef dataframes for each model    
model_coef <- model_coef_nested %>%unnest(coef)
model_coef
# Plot a histogram of the coefficient estimates for year         
model_coef %>% filter(term == "year") %>% ggplot(aes(x = estimate)) +
  geom_histogram()

#Evaluating the fit of many models 
# Extract the fit statistics of each model into dataframes
model_perf_nested <- gap_models %>% mutate(fit = map(model, ~glance(.x)))

# Simplify the fit dataframes for each model    
model_perf <- model_perf_nested %>% unnest(fit)

# Look at the first six rows of model_perf
head(model_perf)
# Extract the statistics of the algeria_model as a dataframe
glance(algeria_model)

#The augment() function can help you explore this fit by appending the predictions to the original data.
# Build the augmented dataframe
algeria_fitted <- augment(algeria_model)

# Compare the predicted values with the actual values of life expectancy
algeria_fitted %>% 
  ggplot(aes(x = year)) +
  geom_point(aes(y = life_expectancy)) + 
  geom_line(aes(y = .fitted), color = "red") #the fit is not the best

#Evaluating the fit of many models
# Extract the fit statistics of each model into dataframes
model_perf_nested <- gap_models %>% 
  mutate(fit = map(model, ~glance(.x)))

# Simplify the fit dataframes for each model    
model_perf <- model_perf_nested %>% 
  unnest(fit)
# Look at the first six rows of model_perf
head(model_perf)

# Plot a histogram of rsquared for the 77 models    
model_perf %>% ggplot(aes(x = r.squared)) + 
  geom_histogram()  

# Extract the 4 best fitting models
best_fit <- model_perf %>% top_n(n = 4, wt = r.squared)
# Extract the 4 models with the worst fit
worst_fit <- model_perf %>% top_n(n = 4, wt = -r.squared)

##Visually inspect the fit of many models
best_augmented <- best_fit %>% 
  # Build the augmented dataframe for each country model
  mutate(augmented = map(model, ~augment(.x))) %>% 
  # Expand the augmented dataframes
  unnest(augmented)

# Compare the predicted values with the actual values of life expectancy 
# for the top 4 best fitting models
best_augmented %>% 
  ggplot(aes(x = year)) +
  geom_point(aes(y = life_expectancy)) + 
  geom_line(aes(y = .fitted), color = "red") +
  facet_wrap(~country, scales = "free_y")

worst_augmented <- worst_fit %>% 
  # Build the augmented dataframe for each country model
  mutate(augmented = map(model, ~augment(.x))) %>% 
  # Expand the augmented dataframes
  unnest(augmented)
# Compare the predicted values with the actual values of life expectancy 
# for the top 4 worst fitting models
worst_augmented %>% 
  ggplot(aes(x = year)) +
  geom_point(aes(y = life_expectancy)) + 
  geom_line(aes(y = .fitted), color = "red") +
  facet_wrap(~country, scales = "free_y") #the model needs to be improved

#Improve the fit of your models - mutiple regression
# Build a linear model for each country using all features
gap_fullmodel <- gap_nested %>% 
  mutate(model = map(data, ~lm(life_expectancy~., data = .x)))

fullmodel_perf <- gap_fullmodel %>% 
  # Extract the fit statistics of each model into dataframes
  mutate(fit = map(model, ~glance(.x))) %>% 
  # Simplify the fit dataframes for each model
  unnest(fit)

# View the performance for the four countries with the worst fitting 
# four simple models you looked at before
fullmodel_perf %>% filter(country %in% worst_fit$country) %>% 
  select(country, adj.r.squared) #performance of each of the four worst performing models based on their adjusted R2 drastically improved 

#important about R2:  While the adjusted R2 does tell us how well the model fit our data, it does not give any indication on how it would perform on new data

##########################################################
##Training, test and validation splits
#How well would my model perform on new (test) data?
#Did I select the best performing model?

library(rsample)
#Split your data into 75% training and 25% testing using the initial_split() function and assign it to gap_split.
#TRAINING_ TO BUILD A MODEL
#TEST _ To evaluate the performance

set.seed(42)

# Prepare the initial split object
gap_split <- initial_split(gapminder, prop = 0.75)
# Extract the training dataframe
training_data <- training(gap_split)
# Extract the testing dataframe
testing_data <- testing(gap_split)
# Calculate the dimensions of both training_data and testing_data
dim(training_data)
dim(testing_data)


#split the training data into a series of 5 train-validate sets using the vfold_cv() function from the rsample package.
set.seed(42)

# Prepare the dataframe containing the cross validation partitions
cv_split <- vfold_cv(training_data, v = 5) #so we will build 5 models

cv_data <- cv_split %>% 
  mutate(
    # Extract the train dataframe for each split
    train = map(splits, ~training(.x)), 
    # Extract the validate dataframe for each split
    validate = map(splits, ~testing(.x))
  )

# Use head() to preview cv_data
head(cv_data)

#build a models on a training dataset
cv_models_lm <- cv_data %>% 
  mutate(model = map(train, ~lm(formula = life_expectancy~., data = .x)))

#Preparing for evaluation
cv_prep_lm <- cv_models_lm %>% 
  mutate(
    # Extract the recorded life expectancy for the records in the validate dataframes
    validate_actual = map(validate, ~.x$life_expectancy),
    # Predict life expectancy for each validate set using its corresponding model
    validate_predicted = map2(.x = model, .y = validate, ~predict(.x, .y))
  )

#evaluate model performance with MAE
library(Metrics)
# Calculate the mean absolute error for each validate fold       
cv_eval_lm <- cv_prep_lm %>% 
  mutate(validate_mae = map2_dbl( validate_actual, validate_predicted, ~mae(actual = .x, predicted = .y)))

# Print the validate_mae column
cv_eval_lm$validate_mae

# Calculate the mean of validate_mae based on 5 train-validate splits
mean(cv_eval_lm$validate_mae) #he predictions of the models are on average off by 1.47 years column. Model predicitions will be off on avarage  for 1.5y

########################RANDOM FOREST
#lets use random-forest model
library(ranger)
#Use ranger() to build a random forest predicting life_expectancy using all features in train for each cross validation partition.
## Build a random forest model for each fold
cv_models_rf <- cv_data %>% 
  mutate(model = map(train, ~ranger(formula = life_expectancy~., data =.x,
                                    num.trees = 100, seed = 42)))

#Add a new column validate_predicted predicting the life_expectancy for the observations in validate using the random forest models you just created.
# Generate predictions using the random forest model
cv_prep_rf <- cv_models_rf %>% 
  mutate(validate_predicted = map2(.x = model, .y = validate, ~predict(.x, .y)$predictions))

#Evaluate a random forest model
cv_eval_rf <- cv_prep_rf %>% 
  mutate(validate_mae = map2_dbl(validate_actual,validate_predicted, ~mae(actual = .x, predicted = .y)))

# Print the validate_mae column
cv_eval_rf$validate_mae
# Calculate the mean of validate_mae column
mean(cv_eval_rf$validate_mae) #You've dropped the average error of your predictions from 1.47 to 0.79!! Much better than the regresion

#Fine tune your random forest model
#vary the mtry parameter (mx number is the number of the features we have 5 i.e. year, population, fertility etc) when building your random forest models on your train data

# Prepare for tuning your cross validation folds by varying mtry
cv_tune <- cv_data %>% 
  crossing(mtry = 2:5) 

# Build a model for each fold & mtry combination
cv_model_tunerf <- cv_tune %>% 
  mutate(model = map2(.x = train, .y = mtry, ~ranger(formula = life_expectancy~., data = .x, mtry = .y, num.trees = 100, seed = 42))) # You've built a model for each fold/mtry combination.


#measure the performance of each mtry value across the 5 cross validation partitions to see if you can improve the model.
#MAE you calculated two exercises ago of 0.795 was for the default mtry value of 2.

# Generate validate predictions for each model
cv_prep_tunerf <- cv_model_tunerf %>% 
  mutate(validate_predicted = map2(.x = model, .y = validate, ~predict(.x, .y)$predictions))

# Calculate validate MAE for each fold and mtry combination
cv_eval_tunerf <- cv_prep_tunerf %>% 
  mutate(validate_mae = map2_dbl(.x = validate_actual, .y = validate_predicted, ~mae(actual = .x, predicted = .y)))

# Calculate the mean validate_mae for each mtry used  
cv_eval_tunerf %>% 
  group_by(mtry) %>% 
  summarise(mean_mae = mean(validate_mae)) 
# Assuming that you've finished your model selection you can conclude that your final (best performing) model will be the random forest model built using ranger with an mtry = 4 and num.trees = 100
#the best was mytr=4

###FINAL: Measuring the test performance (ON TEST DATA)
#Build & evaluate the best model
#bulid a model using ALL training data and test on test data
# Build the model using all training data and the best performing parameter
best_model <- ranger(formula = life_expectancy~., data = training_data, mtry = 4, num.trees = 100, seed = 42)

# Build the model using all training data and the best performing parameter
best_model <- ranger(formula = life_expectancy~., data = training_data,
                     mtry = 4, num.trees = 100, seed = 42)

# Prepare the test_actual vector
test_actual <- testing_data$life_expectancy

# Predict life_expectancy for the testing_data
test_predicted <- predict(best_model, testing_data)$predictions

# Calculate the test MAE
mae(test_actual, test_predicted) #You can claim that based on the test holdout you can expect that your predictions on new data will only be off by a magnitude of 0.679 years.



##############LOGISTIC REGRESSION MODELS (AKA BINARY CLASSIFICATION)
attrition <- readRDS( "attrition.rds" )
set.seed(42)

# Prepare the initial split object
data_split <- initial_split(attrition, prop = 0.75)
# Extract the training dataframe
training_data <- training(data_split)
# Extract the testing dataframe
testing_data <- testing(data_split)
#Build a dataframe for 5-fold cross validation from the training_data using vfold_cv()
set.seed(42)
cv_split <- vfold_cv(training_data, v = 5)
#Prepare the cv_data dataframe by extracting the train and validate dataframes for each fold
cv_data <- cv_split %>% 
  mutate(
    # Extract the train dataframe for each split
    train = map(splits, ~training(.x)),
    # Extract the validate dataframe for each split
    validate = map(splits, ~testing(.x))
  )

#build logistic regression models for each fold in your cross-validation.
# Build a model using the train data for each fold of the cross validation
cv_models_lr <- cv_data %>% 
  mutate(model = map(train, ~glm(formula = Attrition~., data = .x, family = "binomial")))

#####Calculation of the performance for a single model
#To calculate the performance of a classification model you need to compare the actual values of Attrition to those predicted by the model. 
#When calculating metrics for binary classification tasks (such as precision and recall), the actual and predicted vectors must be converted to binary values.

#Extract the model and the validate dataframe from the first fold of the cross-validation.
model <- cv_models_lr$model[[1]]
validate <- cv_models_lr$validate[[1]]

#Extract the Attrition column from the validate dataframe and convert the values to binary (TRUE/FALSE).
validate_actual <- validate$Attrition == "Yes"
validate_actual

#Use model to predict the probabilities of attrition for the validate dataframe
validate_prob <- predict(model, validate, type = "response")
#Convert the predicted probabilities to a binary vector, assume all probabilities greater than 0.5 are TRUE.
validate_predicted <- validate_prob > 0.5
#Use table() to compare the validate_actual and validate_predicted values for the example model and validate dataframe.
library(Metrics)

# Compare the actual & predicted performance visually using a table
table(validate_actual, validate_predicted)
# Calculate the accuracy
accuracy(validate_actual, validate_predicted) #how well your model predicts TRUE nad FALSE classes
# Calculate the precision
precision(validate_actual, validate_predicted) #how often model is accurate in predicting the TRUE class
# Calculate the recall
recall(validate_actual, validate_predicted) #measurees the rate at which model captures the TRUE class - from employees that quit the model was able to capture around 50$% of them

#####Calculation of the performance for all the folds in cross-validation dataframe
#For validate_actual, you can use map() to both extract the Attrition column and convert it to a binary column where "Yes" is converted to TRUE
#For validate_predicted, you need to map() over both the model and validate columns and convert it to a binary column where all probabilities greater than 0.5 are converted to TRUE
cv_prep_lr <- cv_models_lr %>% 
  mutate(
    # Prepare binary vector of actual Attrition values in validate
    validate_actual = map(validate, ~.x$Attrition == "Yes"),
    # Prepare binary vector of predicted Attrition values for validate
    validate_predicted = map2(.x = model, .y = validate, ~predict(.x, .y, type = "response") > 0.5)
  )

# you want to use this model to identify employees that are predicted to leave the company. 
#Ideally, you want a model that can capture as many of the ready-to-leave employees as possible so that you can intervene. 
#The corresponding metric that captures this is the recall metric. 


#As such, you will exclusively use recall to optimize and select your models
# Calculate the validate recall for each cross validation fold
cv_perf_recall <- cv_prep_lr %>% 
  mutate(validate_recall = map2_dbl(validate_actual, validate_predicted, 
                                    ~recall(actual = .x, predicted = .y)))

# Print the validate_recall column
cv_perf_recall$validate_recall

# Calculate the average of the validate_recall column
mean(cv_perf_recall$validate_recall) #bas i nije neka vrijednost

######################RANDOM FOREST FOR CALSSIFICATION
# Prepare for tuning your cross validation folds by varying mtry
cv_tune <- cv_data %>%crossing(mtry = c(2, 4, 8, 16)) 

# Build a cross validation model for each fold & mtry combination
cv_models_rf <- cv_tune %>% mutate(model = map2(train, mtry, ~ranger(formula = Attrition~., 
                                           data = .x, mtry = .y,
                                           num.trees = 100, seed = 42)))

cv_prep_rf <- cv_models_rf %>% 
  mutate(
    # Prepare binary vector of actual Attrition values in validate
    validate_actual = map(validate, ~.x$Attrition == "Yes"),
    # Prepare binary vector of predicted Attrition values for validate
    validate_predicted = map2(.x = model, .y = validate, ~predict(.x, .y, type = "response")$predictions == "Yes")
  )

# Calculate the validate recall for each cross validation fold
cv_perf_recall <- cv_prep_rf %>% 
  mutate(recall = map2_dbl(validate_actual, validate_predicted,  ~recall(actual = .x, predicted = .y)))

# Calculate the mean recall for each mtry used  
cv_perf_recall %>% 
  group_by(mtry) %>% 
  summarise(mean_recall = mean(recall )) # none of the random forest models were able to outperform the logistic regression model with respect to recall.

####################### Build final classification model
# Build the logistic regression model using all training data
best_model <- glm(formula = Attrition~., data = training_data, family = "binomial")


# Prepare binary vector of actual Attrition values for testing_data
test_actual <- testing_data$Attrition == "Yes"

# Prepare binary vector of predicted Attrition values for testing_data
test_predicted <- predict(best_model, testing_data, type = "response") > 0.5

# Compare the actual & predicted performance visually using a table
table(test_actual, test_predicted)

# Calculate the test accuracy
accuracy(test_actual, test_predicted)

# Calculate the test precision
precision(test_actual, test_predicted)

# Calculate the test recall
recall(test_actual, test_predicted) #You now have a model that you can expect to identify 36% of employees that are at risk to leave the organization