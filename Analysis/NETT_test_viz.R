# NETT_test_viz.R

# Before running this script, you need to run merge_csvs to merge all of the agents'
# output into a single, standardized format dataframe for training and test data

# Variables --------------------------------------------------------------------

# USER-SPECIFIED VARIABLES
data_loc <- "/Users/samanthawood/Documents/WoodLab/AI Papers/EmbodiedPipeline/segmentation_data.R"
key_csv <- "/Users/samanthawood/Documents/WoodLab/AI Papers/EmbodiedPipeline/segmentation_key.csv"
results_wd <- "/Users/samanthawood/Documents/WoodLab/AI Papers/EmbodiedPipeline/"


# Set Up -----------------------------------------------------------------------

library(tidyverse)
library(scales)

cond_key <- read.csv(key_csv)
cond_key <- select(cond_key, -c(left, right))

# Load test data
load(data_loc)
rm(train_data)

# Add test conditions from the key
test_data <- test_data %>%
  mutate(left_right = paste(left.monitor, right.monitor, sep="-")) %>%
  left_join(cond_key, 
            by=c("left_right" = "left_right", "imprinting" = "imprinting"))
# Code each episode correct/incorrect
test_data <-test_data %>%
  mutate(correct_steps = if_else(correct_monitor == "left", left_steps, right_steps)) %>%
  mutate(incorrect_steps = if_else(correct_monitor == "left", right_steps, left_steps)) %>%
  mutate(percent_correct = correct_steps / (correct_steps + incorrect_steps)) %>%
  mutate()

setwd(results_wd)


# Plot aesthetic settings ------------------------------------------------------
p <- ggplot() + 
  theme_classic() + 
  theme(axis.text.x = element_text(size = 6)) + 
  ylab("Percent Correct Steps") + 
  scale_y_continuous(expand = c(0, 0), limits = c(0, 1), breaks=seq(0,1,.1), labels = scales::percent) +
  scale_x_discrete(labels = label_wrap(25)) +
  geom_hline(yintercept = .5)


# Plot by agent ----------------------------------------------------------------

## Group data by test conditions
by_test_cond <- test_data %>%
  group_by(imprinting, agent, cond_name) %>%
  summarise(avgs = mean(percent_correct, na.rm = TRUE), 
            sd = sd(percent_correct, na.rm = TRUE), 
            count = length(percent_correct)) %>%
  mutate(se = sd / sqrt(count)) %>%
  mutate(imp_agent = paste(imprinting, agent, sep="_"))

for (i in unique(by_test_cond$imp_agent))
{
  bar_data <- by_test_cond %>%
    filter(imp_agent == i)
  
  p + 
    geom_col(data = bar_data, aes(x=cond_name, y = avgs)) + 
    geom_errorbar(data = bar_data, aes(x = cond_name, 
                                       ymin = avgs - se, 
                                       ymax = avgs + se))
    
  img_name <- paste0(i, "_test.png")
  ggsave(img_name)
}


# Plot by imprinting condition -------------------------------------------------
by_imp_cond <- by_test_cond %>%
  ungroup() %>%
  group_by(imprinting, cond_name) %>%
  summarise(avgs_by_imp = mean(avgs, na.rm = TRUE), 
            sd = sd(avgs, na.rm = TRUE), 
            count = length(avgs)) %>%
  mutate(se = sd / sqrt(count))

for (i in unique(by_imp_cond$imprinting))
{
  bar_data <- by_imp_cond %>%
    filter(imprinting == i)
  p + 
    geom_col(data = bar_data, aes(x=cond_name, y = avgs_by_imp)) + 
    geom_errorbar(data = bar_data, aes(x = cond_name, 
                                       ymin = avgs_by_imp - se, 
                                       ymax = avgs_by_imp + se))
    
  
  img_name <- paste0(i, "_test.png")
  ggsave(img_name)
}


# Plot across all imprinting conditions ----------------------------------------
by_test_cond <- by_test_cond %>%
  ungroup() %>%
  group_by(cond_name) %>%
  summarise(all_avgs = mean(avgs, na.rm = TRUE), 
            sd = sd(avgs, na.rm = TRUE), 
            count = length(avgs)) %>%
  mutate(se = sd / sqrt(count))

p + 
  geom_col(data = by_test_cond, aes(x=cond_name, y = all_avgs)) + 
  geom_errorbar(data = by_test_cond, aes(x = cond_name, 
                                     ymin = all_avgs - se, 
                                     ymax = all_avgs + se))
ggsave("all_imprinting_conds_test.png")