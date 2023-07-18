# merge_csvs.R

# For a specified directory (see below), takes all of the csv files
# and compiles them into a single data file

# NOTE: For ease of use across many different experimental designs, 
# this script assumes that all files use a common naming scheme with the 
# following criteria:
# 1) The conditions are specified at the beginning of the filename
# 2) The conditions are followed by a dash
# 3) There are no other dashes in the name
# 4) The agent ID number is the only number in the file name
# 5) The filename ends with either train.csv or test.csv
# for example "fork_side-agent3_train.csv"


# Variables --------------------------------------------------------------------

# USER-SPECIFIED VARIABLES
data_wd <- "/home/mchivuku/projects/embodied_pipeline/benchmark_experiments/data/parsing_analysis/"
results_wd <- "/home/mchivuku/projects/embodied_pipeline/benchmark_experiments/data/parsing_analysis/"
results_name <- "segmentation_data.R"
# If you don't want to save a csv of the train and/or test results, set to NULL
csv_train_name <- NULL
csv_test_name <- "test.csv"

# Set Zones:
upper_x_lim <- 10
lower_x_lim <- -10
one_third <- (upper_x_lim - lower_x_lim)/3
lower_bound <- lower_x_lim + one_third
upper_bound <- upper_x_lim - one_third


# Set Up -----------------------------------------------------------------------

# Import libraries
library(tidyverse)

# Get all of the subdirectory csv filenames
setwd(data_wd)
train_files <- list.files(pattern="train.csv", recursive = TRUE)
test_files <- list.files(pattern="exp.csv", recursive = TRUE)

# Main Function ----------------------------------------------------------------

# This function reads in a single csv (later we'll lapply it across all files)
read_data <- function(filename)
{
  # Read the csv file
  data <- read.csv(filename)
  
  # Summarize by zones
  data <- data %>%
    mutate(left = case_when( agent.x < lower_bound ~ 1, agent.x >= lower_bound ~ 0)) %>%
    mutate(right = case_when( agent.x > upper_bound ~ 1, agent.x <= upper_bound ~ 0)) %>%
    mutate(middle = 1- left - right)
  # Quick check to make sure that one and only one zone is chosen at each step
  stopifnot(all( (data$left + data$right + data$middle == 1) ))
  # Summarize at the episode level
  data <- data %>%
    group_by(Episode, left.monitor, right.monitor, correct.monitor) %>%
    summarise(left_steps = sum(left), 
              right_steps = sum(right), 
              middle_steps = sum(middle)) %>%
    mutate(Episode = as.numeric(Episode)) %>%
    mutate(left.monitor = sub(" ", "", left.monitor)) %>%
    mutate(right.monitor = sub(" ", "", right.monitor)) %>%
    mutate(correct.monitor = sub(" ", "", correct.monitor)) %>%
    ungroup()
  
  # Add columns for original filename, agent ID number, and imprinting condition
  data$filename <- basename(filename)
  data$agent <- gsub("\\D", "", data$filename)
  data$imprinting <- strsplit(basename(filename), "-")[[1]][1]
  print(strsplit(basename(filename), "-")[[1]][1],zero.print = ".")
  return(data)
}

# Combine csv's and save results -----------------------------------------------

# Combine all the training
train_data <- lapply(train_files, FUN = read_data)
train_data <- bind_rows(train_data)

# Combine all the testing
test_data <- lapply(test_files, FUN = read_data)
test_data <- bind_rows(test_data)

# Save it
setwd(results_wd)
save(train_data, test_data, file=results_name)
if( !is.null(csv_train_name) ) write.csv(train_data, csv_train_name)
if( !is.null(csv_test_name) ) write.csv(test_data, csv_test_name)