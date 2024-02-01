library(tidyverse)
library(ggplot2)

abia_data = read.csv('ABIA.csv', header = TRUE)

# Converting DepTime to hour of the day
abia_data$DepHour = floor(abia_data$DepTime / 100)

# Calculate average departure delays by airline and hour
avg_delays = abia_data %>%
  group_by(UniqueCarrier, DepHour) %>%
  summarize(AvgDepDelay = mean(DepDelay, na.rm = TRUE))

# Plotting average departure delay by airline across different hours of the day 
ggplot(avg_delays, aes(x = DepHour, y = AvgDepDelay, color = UniqueCarrier)) +
  geom_line() +
  geom_point() +
  facet_wrap(~UniqueCarrier, scales = "fixed", ncol = 4) + 
  theme_minimal() +
  labs(title = "Average Departure Delays by Airline and Hour of Day",
       x = "Hour of Day",
       y = "Average Delay in minutes") +
  scale_x_continuous(breaks = seq(0, 24, by = 6)) + 
  theme(legend.position = "none") 

