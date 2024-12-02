# Libraries
library(tidymodels)
library(tidyverse)
library(vroom)
library(stacks)
library(doParallel)

# find number of cores
num_cores <- 4

# Additional data
inflation <- c(1.6, 2.0, 1.5, 1.1, 1.4, 1.8, 2.0, 1.5, 1.2, 1.0, 1.2, 1.5,
               1.6, 1.1, 1.5, 2.0, 2.1, 2.1, 2.0, 1.7, 1.7, 1.7, 1.3, 0.8,
               -0.1, 0.0, -0.1, -0.2, 0.0, 0.1, 0.2, 0.2, 0.0, 0.2, 0.5, 0.7,
               1.4, 1.0, 0.9, 1.1, 1.0, 1.0, 0.8, 1.1, 1.5, 1.6, 1.7, 2.1,
               2.5, 2.7, 2.4, 2.2, 1.9, 1.6, 1.7, 1.9, 2.2, 2.0, 2.2, 2.1,
               2.1, 2.2, 2.4)
inf_df <- tibble(date = seq(as.Date("2013/1/1"),
                            as.Date("2018/3/1"),
                            "month"),
                 rate = inflation) %>%
  mutate(year = year(date)) %>%
  mutate(month = month(date)) %>%
  select(!date)

# Load in data
train_dirty <- vroom("train.csv") %>%
  left_join(inf_df, by = c("year", "month"))
test_dirty <- vroom("test.csv") %>%
  left_join(inf_df, by = c("year", "month"))

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
    prophet_cv %>%
      modeltime_forecast(new_data = testing(folds),
                         actual_data = training(folds))
    stopCluster(cl)

    # Fit model and make predictions
    prophet_fit <- prophet_cv %>%
      modeltime_refit(data = tmp_train)
    prophet_predictions <- predict(prophet_fit,
                                   new_data = tmp_test)$.pred

    # Put predictions in output data frame
    output[(output$store == store)
           & (output$item == item), "sales"] <- prophet_predictions

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

