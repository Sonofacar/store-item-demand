# Libraries
library(tidymodels)
library(tidyverse)
library(vroom)
library(stacks)
library(doParallel)

# find number of cores
num_cores <- parallel::detectCores()

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
holidays <- c("ChristmasDay",
              "ChristmasEve",
              "Easter",
              "EasterMonday",
              "EasterSunday",
              "LaborDay",
              "NewYearsDay",
              "DENewYearsEve",
              "USColumbusDay",
              "USIndependenceDay",
              "USMemorialDay",
              "USMLKingsBirthday",
              "USThanksgivingDay")

# Load in data
train_dirty <- vroom("train.csv") %>%
  mutate(year = year(date),
         month = month(date)) %>%
  left_join(inf_df, by = c("year", "month")) %>%
  select(!year) %>%
  select(!month)
test_dirty <- vroom("test.csv") %>%
  mutate(year = year(date),
         month = month(date)) %>%
  left_join(inf_df, by = c("year", "month")) %>%
  select(!year) %>%
  select(!month)

# Create the recipe
recipe <- recipe(sales ~ ., train_dirty) %>%
  step_holiday(date, holidays = holidays) %>%
  step_date(date, features = "decimal") %>%
  step_date(date, features = "dow", label = FALSE) %>%
  step_mutate(date_dow = factor(date_dow)) %>%
  step_dummy(date_dow) %>%
  step_mutate(sinYear = sin(date_decimal * pi * 2),
              cosYear = cos(date_decimal * pi * 2)) %>%
  step_rm(date, store, item) %>%
  step_normalize(all_predictors())

# Set up output
input <- train_dirty
output <- test_dirty

# Create the model and workflow
penalized_model <- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("regression")
penalized_workflow <- workflow() %>%
  add_model(penalized_model) %>%
  add_recipe(recipe)

# Fit a penalized regression for each store-item combination
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
    folds <- vfold_cv(tmp_train, v = 10, repeats = 1)

    # Tune
    cl <- makePSOCKcluster(num_cores)
    registerDoParallel(cl)
    param_grid <- grid_regular(penalty(),
                               mixture(),
                               levels = 10)
    penalized_cv <- penalized_workflow %>%
      tune_grid(resamples = folds,
                grid = param_grid,
                metrics = metric_set(smape),
                control = control_stack_grid())
    best_tune <- penalized_cv %>%
      select_best(metric = "smape")
    stopCluster(cl)

    # Fit model and make predictions
    penalized_fit <- penalized_workflow %>%
      finalize_workflow(best_tune) %>%
      fit(data = tmp_train)
    penalized_predictions <- predict(penalized_fit,
                                     new_data = tmp_test)$.pred
    train_predictions <- predict(penalized_fit,
                                 new_data = tmp_train)$.pred

    # Put training predictions in the original dataset
    tmp_train$pred <- train_predictions
    input <- left_join(input,
                       tmp_train[c("date", "store", "item", "pred")],
                       by = c("date", "store", "item"))

    # Put predictions in output data frame
    tmp_test$sales <- penalized_predictions
    output <- left_join(output, tmp_test[c("id", "sales")], by = "id")
  }
}

# print output
vroom_write(output[c("id", "sales")], "submission.csv", delim = ",")

