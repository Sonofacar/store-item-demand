# Libraries
library(tidymodels)
library(tidyverse)
library(vroom)
library(stacks)
library(doParallel)
library(modeltime)
library(timetk)

# find number of cores
num_cores <- 4

# Load in data
train_dirty <- vroom("../input/demand-forecasting-kernels-only/train.csv")
test_dirty <- vroom("../input/demand-forecasting-kernels-only/test.csv")

# Set up output
output <- test_dirty
output$sales <- 0

# Fit a prophet regression for each store-item combination
count <- 1
stores <- 1:10
items <- 1:50
for (store in stores) {
  for (item in items) {
    # Get data segments
    tmp_train <- train_dirty[(train_dirty$store == store)
                             & (train_dirty$item == item), ]
    tmp_test <- test_dirty[(test_dirty$store == store)
                           & (test_dirty$item == item), ]

    # Create folds
    folds <- time_series_split(tmp_train,
                               assess = "3 months",
                               cumulative = TRUE)

    # Create the model
    prophet_model <- prophet_reg() %>%
      set_engine("prophet") %>%
      fit(sales ~ ., data = training(folds))

    # Tune
    cl <- makePSOCKcluster(num_cores)
    registerDoParallel(cl)
    prophet_cv <- modeltime_calibrate(prophet_model,
                                      new_data = testing(folds))
    stopCluster(cl)

    # Fit model and make predictions
    prophet_fit <- prophet_cv %>%
      modeltime_refit(data = tmp_train)
    prophet_predictions <- prophet_fit %>%
      modeltime_forecast(new_data = tmp_test)

    # Put predictions in output data frame
    output[(output$store == store)
           & (output$item == item), "sales"] <- prophet_predictions$.value

    # Print to monitor progress
    paste("Store: ", store, "\t",
          "Item: ", item, "   \t",
          "Count: ", count, "/500", sep = "") %>%
      write(stdout())
    count <- count + 1
  }
}

# print output
vroom_write(output[c("id", "sales")], "submission.csv", delim = ",")

