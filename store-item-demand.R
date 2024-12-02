library(tidymodels)
library(tidyverse)
library(vroom)
library(stacks)
library(doParallel)
library(modeltime)
library(timetk)
library(modeltime.gluonts)

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
prepped_recipe <- prep(recipe)
clean_data <- bake(prepped_recipe, new_data = train_dirty)

store <- sample(train_dirty$store, 1)
item <- sample(train_dirty$item, 1)
tmp_train <- train_dirty[(train_dirty$store == store)
                         & (train_dirty$item == item), ]
tmp_test <- test_dirty[(test_dirty$store == store)
                       & (test_dirty$item == item), ]

folds <- vfold_cv(tmp_train, v = 10, repeats = 1)

#################################
# Penalized Logistic Regression #
#################################

# Create the model
penalized_model <- linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet") %>%
  set_mode("regression")

# Create the workflow
penalized_workflow <- workflow() %>%
  add_model(penalized_model) %>%
  add_recipe(recipe)

# Cross validate
param_grid <- grid_regular(penalty(),
                           mixture(),
                           levels = 20)
penalized_cv <- penalized_workflow %>%
  tune_grid(resamples = folds,
            grid = param_grid,
            metrics = metric_set(smape),
            control = control_stack_grid())
show_best(penalized_cv, metric = "smape")
best_tune <- penalized_cv %>%
  select_best(metric = "smape")

penalized_fit <- penalized_workflow %>%
  finalize_workflow(best_tune) %>%
  fit(data = tmp_train)
penalized_predictions <- predict(penalized_fit,
                                 new_data = tmp_train)$.pred

# Fit model and make predictions
penalized_fit <- penalized_workflow %>%
  finalize_workflow(best_tune) %>%
  fit(data = tmp_train)
penalized_predictions <- predict(penalized_fit,
                                 new_data = tmp_test)$.pred
#########
# ARIMA #
#########

# Create time series splits
arima_folds <- time_series_split(tmp_train,
                                 assess = "3 months",
                                 cumulative = TRUE)
arima_folds %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(date, sales, .interactive = FALSE)

# ARIMA specific recipe
arima_recipe <- recipe(sales ~ ., data = train_dirty)

# Create the model
arima_model <- arima_reg(seasonal_period = 365,
                         seasonal_ar = 365,
                         seasonal_ma = 5,
                         non_seasonal_ar = 5,
                         non_seasonal_ma = 5,
                         non_seasonal_differences = 5,
                         seasonal_differences = 5) %>%
  set_engine("auto_arima")

# Create the workflow
arima_workflow <- workflow() %>%
  add_model(arima_model) %>%
  add_recipe(arima_recipe) %>%
  fit(data = training(arima_folds))

# Cross validate
arima_cv <- modeltime_calibrate(arima_workflow,
                                new_data = testing(arima_folds))

# Visualize CV
p1 <- arima_cv %>%
  modeltime_forecast(new_data = testing(arima_folds),
                     actual_data = training(arima_folds)) %>%
  plot_modeltime_forecast(.interactive = FALSE)

# Finalize fit
arima_fit <- arima_cv %>%
  modeltime_refit(data = tmp_train)

# Visualize fit
p2 <- arima_fit %>%
  modeltime_forecast(new_data = tmp_test,
                     actual_data = tmp_train) %>%
  plot_modeltime_forecast(.interactive = FALSE)

###########
# Prophet #
###########

# Create folds
prophet_folds <- time_series_split(tmp_train,
                                   assess = "3 months",
                                   cumulative = TRUE)
# Create the model
prophet_model <- prophet_reg() %>%
  set_engine("prophet") %>%
  fit(sales ~ date, data = training(prophet_folds))

# Calibrate the model
prophet_cv <- modeltime_calibrate(prophet_model,
                                  new_data = testing(prophet_folds))

# Visualize CV
prophet_cv %>%
  modeltime_forecast(new_data = testing(prophet_folds),
                     actual_data = training(prophet_folds)) %>%
  plot_modeltime_forecast(.interactive = FALSE)

# Finalize fit
prophet_fit <- prophet_cv %>%
  modeltime_refit(data = tmp_train)

# Visualize fit
prophet_fit %>%
  modeltime_forecast(new_data = tmp_test,
                     actual_data = tmp_train) %>%
  plot_modeltime_forecast(.interactive = FALSE)

###########
# Gluonts #
###########

# Create an id column
tmp_train <- train_dirty %>%
  filter(store == 4) %>%
  filter(item < 10) %>%
  mutate(id = paste(store, item, sep = "_")) %>%
  select(id, date, sales)
tmp_test <- test_dirty %>%
  filter(store == 4) %>%
  filter(item < 10) %>%
  mutate(id = paste(store, item, sep = "_")) %>%
  select(id, date)


# Fold data
gluonts_folds <- time_series_split(tmp_train,
                                   assess = "3 months",
                                   cumulative = TRUE)

# Create the model
gluonts_model <- deep_ar(id = "id",
                         freq = "D",
                         prediction_length = 90) %>%
  set_engine("gluonts_deepar") %>%
  fit(sales ~ ., data = training(gluonts_folds))

# Calibrate the model
gluonts_cv <- gluonts_model %>%
  modeltime_table() %>%
  modeltime_calibrate(new_data = testing(gluonts_folds), quiet = FALSE)

# Visualize CV
gluonts_cv %>%
  modeltime_forecast(new_data = testing(gluonts_folds),
                     actual_data = gluonts_folds) %>%
  plot_modeltime_forecast(.interactive = FALSE)

