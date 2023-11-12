#!/usr/bin/env Rscript

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

# Read in the user-specified variables:
library(argparse)
parser <- ArgumentParser(description="An executable R script for the Newborn Embodied Turing Tests to merge the log files across many agents")
parser$add_argument("--logs-dir", type="character", dest="logs_dir",
                    help="Working directory of the agents' log files",
                    required=TRUE)
parser$add_argument("--results-dir", type="character", dest="results_dir",
                    help="Working directory to store the merged output",
                    required=TRUE)
parser$add_argument("--results-name", type="character", dest="results_name",
                    help="File name for the R file storing the results",
                    required=TRUE)
parser$add_argument("--csv-train", type="character", dest="csv_train_name",
                    help="File name for the csv file storing the training results",
                    required=FALSE)
parser$add_argument("--csv-test", type="character", dest="csv_test_name",
                    help="File name for the csv file storing the testing results",
                    required=FALSE)
args <- parser$parse_args()
data_wd <- args$logs_dir; results_wd <- args$results_dir; results_name <- args$results_name
csv_train_name <- args$csv_train_name; csv_test_name <- args$csv_test_name

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
train_files <- list.files(pattern="train.csv", recursive = FALSE)
test_files <- list.files(pattern="exp.csv", recursive = FALSE)

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
    group_by(Episode, left.monitor, right.monitor) %>%
    summarise(left_steps = sum(left), 
              right_steps = sum(right), 
              middle_steps = sum(middle)) %>%
    mutate(Episode = as.numeric(Episode)) %>%
    mutate(left.monitor = sub(" ", "", left.monitor)) %>%
    mutate(right.monitor = sub(" ", "", right.monitor)) %>%
    ungroup()
  
  # Add columns for original filename, agent ID number, and imprinting condition
  data$filename <- basename(filename)
  data$agent <- gsub("\\D", "", data$filename)
  data$imprinting <- strsplit(filename, "-")[[1]][1]

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




