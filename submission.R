# Libraries
library(tidymodels)
library(tidyverse)
library(vroom)
library(stacks)

# Holidays
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
train_dirty <- vroom("../input/demand-forecasting-kernels-only/train.csv") %>%
  mutate(year = year(date),
         month = month(date)) %>%
  left_join(inf_df, by = c("year", "month")) %>%
  select(!year) %>%
  select(!month)
test_dirty <- vroom("../input/demand-forecasting-kernels-only/test.csv") %>%
  mutate(year = year(date),
         month = month(date)) %>%
  left_join(inf_df, by = c("year", "month")) %>%
  select(!year) %>%
  select(!month)

# Set up output
output <- test_dirty
output$sales <- 0

# Recipe
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
# Model
model <- boost_tree(tree_depth = 2,
                    trees = 1000,
                    learn_rate = 0.01) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

# Workflow
wflow <- workflow() %>%
  add_model(model) %>%
  add_recipe(recipe)

# Fit a prophet regression for each store-item combination
count <- 1
stores <- 1:10
items <- 1:50
for (store in stores) {
  for (item in items) {
    # Get data segments
    tmp_train <- train_dirty %>%
      filter(store == store) %>%
      filter(item == item)
    tmp_test <- test_dirty %>%
      filter(store == store) %>%
      filter(item == item)

    # Fit model
    f <- wflow %>%
      fit(data = tmp_train)
    preds <- predict(f, new_data = tmp_test)$.pred

    # Save output
    output[(output$store == store)
           & (output$item == item), "sales"] <- preds

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

