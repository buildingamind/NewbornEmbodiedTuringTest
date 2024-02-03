#!/usr/bin/env Rscript

# NETT_train_viz.R

# Before running this script, you need to run merge_csvs to merge all of the agents'
# output into a single, standardized format dataframe for training and test data

# Variables --------------------------------------------------------------------

# Read in the user-specified variables:
library(argparse)
parser <- ArgumentParser(description="An executable R script for the Newborn Embodied Turing Tests to analyze test trials")
parser$add_argument("--data-loc", type="character", dest="data_loc",
                    help="Full filename (inc working directory) of the merged R data",
                    required=TRUE)
parser$add_argument("--results-wd", type="character", dest="results_wd",
                    help="Working directory to save the resulting visualizations",
                    required=TRUE)
parser$add_argument("--ep-bucket", type="integer", dest="ep_bucket_size",
                    help="How many episodes to group the x-axis by",
                    required=TRUE)
parser$add_argument("--num-episodes", type="integer", dest="num_episodes",
                    help="How many episodes should be included",
                    required=TRUE)
args <- parser$parse_args()
data_loc <- args$data_loc; results_wd <- args$results_wd; ep_bucket_size <- args$ep_bucket_size; num_episodes <- args$num_episodes

# Set Up -----------------------------------------------------------------------

library(tidyverse)

load(data_loc)
rm(test_data)
setwd(results_wd)

train_data_fixed <- train_data %>%
  filter(Episode < num_episodes) %>%
  # Create variables for correct/incorrect calculations
  mutate(correct_steps = if_else(right.monitor == "White", left_steps, right_steps)) %>%
  mutate(incorrect_steps = if_else(right.monitor == "White", right_steps, left_steps)) %>%
  mutate(percent_correct = correct_steps / (correct_steps + incorrect_steps)) %>%
  # Summarise data by condition, agent, and episode bucket for graphing
  mutate(episode_block = Episode%/%ep_bucket_size + 1) %>%
  group_by(imprint.cond, agent, episode_block) %>%
  summarise(avgs = mean(percent_correct, na.rm = TRUE),
            sd = sd(percent_correct, na.rm = TRUE),
            count = length(percent_correct)) %>%
  mutate(se = sd / sqrt(count)) %>%
  # Convert numerical variables into correct type
  mutate(episode_block = as.numeric(episode_block)) %>%
  mutate(agent = as.numeric(agent)) %>%
  ungroup()


# Plot line graphs by imprinting condition -------------------------------------

for (cond in unique(train_data_fixed$imprint.cond))
{
  data <- train_data_fixed %>%
    filter(imprint.cond == cond)
  
  ggplot(data=data, aes(x=episode_block, y=avgs, color=as.factor(agent))) +
    geom_line() +
    theme_classic(base_size = 16) +
    geom_hline(yintercept = .5, linetype = 2) +
    xlab(sprintf("Groups of %d Episodes", ep_bucket_size)) + 
    ylab("Average Time with Imprinted Object") +
    scale_y_continuous(expand = c(0, 0), limits = c(0, 1), 
                       breaks=seq(0,1,.1), labels = scales::percent) + 
    scale_x_continuous(expand = c(0, 0), limits = c(0, num_episodes/ep_bucket_size), 
                       breaks = seq(0, num_episodes / ep_bucket_size, 1)) +
    theme(legend.position="none") 
  
  img_name <- paste0(cond, "_train.png")
  ggsave(img_name)
}

