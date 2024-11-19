library(tidyverse)
library(forecast)
library(patchwork)
library(vroom)

data <- vroom("train.csv")

stores <- sample(data$store %>% unique, 2)
items <- sample(data$item %>% unique, 2)
colors <- c("blue", "darkgreen")

plot <- NULL
for (i in 1:2) {
  tmp1 <- ggplot(data[(data$store == stores[i]) & (data$item == items[i]), ]) +
    geom_line(aes(x = date, y = sales), color = colors[i]) +
    labs(title = paste("Store: ",
                       stores[i],
                       ",",
                       "Item: ",
                       items[i],
                       sep = "")) +
    theme_classic()
  tmp2 <- data[(data$store == stores[i]) & (data$item == items[i]), ] %>%
    pull(sales) %>%
    ggAcf(lag.max = 31) +
    labs(title = paste("Store: ",
                       stores[i],
                       ",",
                       "Item: ",
                       items[i],
                       sep = "")) +
    theme_classic()
  tmp3 <- data[(data$store == stores[i]) & (data$item == items[i]), ] %>%
    pull(sales) %>%
    ggAcf(lag.max = 2 * 365) +
    labs(title = paste("Store: ",
                       stores[i],
                       ",",
                       "Item: ",
                       items[i],
                       sep = "")) +
    theme_classic()
  plot <- plot / (tmp1 | tmp2 | tmp3)
}

