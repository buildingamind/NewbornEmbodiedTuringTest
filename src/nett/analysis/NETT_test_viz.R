#!/usr/bin/env Rscript

# NETT_test_viz.R

# Before running this script, you need to run merge_csvs to merge all of the agents'
# output into a single, standardized format dataframe for training and test data

# Variables --------------------------------------------------------------------

# Read in the user-specified variables:
library(argparse)
parser <- ArgumentParser(description="An executable R script for the Newborn Embodied Turing Tests to analyze test trials")
parser$add_argument("--data-loc", type="character", dest="data_loc",
                    help="Full filename (inc working directory) of the merged R data",
                    required=TRUE)
parser$add_argument("--chick-file", type="character", dest="chick_file",
                    help="Full filename (inc working directory) of the chick data CSV file",
                    required=TRUE)
parser$add_argument("--results-wd", type="character", dest="results_wd",
                    help="Working directory to save the resulting visualizations",
                    required=TRUE)
parser$add_argument("--bar-order", type = "character", default = "default", dest = "bar_order",
                    help="Order of bars. Use 'default', 'asc', 'desc', or specify indices separated by commas (e.g., '3,2,1,4')",
                    required=FALSE)
parser$add_argument("--color-bars", type = "character", dest="color_bars",
                    help="Should the bars be colored by test condition?",
                    required=TRUE)

# Set script variables based on user input
args <- parser$parse_args()
data_loc <- args$data_loc; chick_file <- args$chick_file; results_wd <- args$results_wd; bar_order <- args$bar_order
if( args$color_bars %in% c("t", "T", "true", "TRUE")) {color_bars <- TRUE} else { color_bars <- FALSE}

# Set Up -----------------------------------------------------------------------

library(tidyverse)
library(stringr)

# Load the chick data
chick_data <- read.csv(chick_file)

# Load test data
load(data_loc)
rm(train_data)

# Code each episode correct/incorrect
test_data <- test_data %>%
  mutate(correct_steps = if_else(correct.monitor == " left", left_steps, right_steps)) %>%
  mutate(incorrect_steps = if_else(correct.monitor == " left", right_steps, left_steps)) %>%
  mutate(percent_correct = correct_steps / (correct_steps + incorrect_steps))

# Adjust bar order according to user input -------------------------------------

# Create a variable to store the final order
order <- NULL
if (bar_order == "default" || bar_order == "asc" || bar_order == "desc"){
  order <- bar_order
}else {
  order <- as.integer(strsplit(order_input, ",")[[1]])
}

# Conditionally reorder the dataframe based on user input
if (!is.null(order)) {
  if (order == "desc") {
    test_data <- test_data %>%
      arrange(desc(percent_correct)) %>%
      mutate(test.cond = factor(test.cond, levels = unique(test.cond)))
  } else if (order == "asc"){
    test_data <- test_data %>%
      arrange(percent_correct) %>%
      mutate(test.cond = factor(test.cond, levels = unique(test.cond)))
  } else if (order != "default") {
    # Map numeric indices to factor levels
    current_order <- levels(factor(test_data$test.cond))
    new_order <- current_order[order]
    test_data$test.cond <- factor(test_data$test.cond, levels = new_order)
  }
  # If order is "default", no need to change anything
}


# Plot aesthetic settings ------------------------------------------------------
custom_palette <- c("#3F8CB7", "#FCEF88", "#5D5797", "#62AC6B", "#B74779", "#2C4E98","#CCCCE7", "#08625B", "#D15056")
chickred <- "#AF264A"

p <- ggplot() +
  theme_classic() +
  theme(axis.text.x = element_text(size = 6)) +
  ylab("Percent Correct") +
  xlab("Test Condition") +
  scale_y_continuous(expand = c(0, 0), limits = c(0, 1), breaks=seq(0,1,.1), labels = scales::percent) +
  geom_hline(yintercept = .5, linetype = 2) +
  scale_fill_manual(values = custom_palette) +
  scale_colour_manual(values = custom_palette) +
  theme(axis.title = element_text(face="bold"),
        axis.text.x = element_text(face="bold", size=8.5),
        axis.text.y = element_text(face="bold", size=8.5))


# Bar Chart Function -----------------------------------------------------------
make_bar_charts <- function(data, dots, aes_y, error_min, error_max, img_name)
{
  p +

    # Add chicken performance FIRST to sort the bars
    geom_errorbar(data=chick_data, width = 0.7, colour = chickred,
                  aes(x=test.cond, ymin=avg, ymax=avg)) +

    # Model performance: bars
    {if(color_bars)geom_col(data = data, width = 0.7, aes(x=test.cond, y = {{aes_y}}, fill = test.cond))}+
    {if(!color_bars)geom_col(data = data, width = 0.7, aes(x=test.cond, y = {{aes_y}}), fill = "gray45")}+
    # Model performance: error bars
    geom_errorbar(data = data, width = 0.3,
                  aes(x = test.cond, ymin = {{error_min}}, ymax = {{error_max}})) +
    # Model performance: dots
    {if(!is.null(dots))geom_jitter(data = dot_data, aes(x=test.cond, y = avgs), width = .3)}+
    theme(legend.position="none") +

    # Add chicken performance again so that it shows up on top
    # Chick performance: lines (errorbar) with ribbons (crossbar)
    geom_errorbar(data=chick_data, width = 0.7, colour = chickred,
                aes(x=test.cond, ymin=avg, ymax=avg)) +
    geom_crossbar(data=chick_data, width = 0.7, 
                  linetype = 0, fill = chickred, alpha = 0.2,
                  aes(x = test.cond, y = avg, 
                      ymin = avg - avg_dev, ymax = avg + avg_dev)) 
  
  ggsave(img_name, width = 6, height = 6)
}

# Switch wd before we save the graphs
setwd(results_wd)

# Plot by agent ----------------------------------------------------------------
## Leave rest data for agent-level graphs

## Group data by test conditions
by_test_cond <- test_data %>%
  group_by(imprint.cond, agent, test.cond) %>%
  summarise(avgs = mean(percent_correct, na.rm = TRUE), 
            sd = sd(percent_correct, na.rm = TRUE), 
            count = length(percent_correct),
            tval = tryCatch({ (t.test(percent_correct, mu=0.5)$statistic)}, error = function(err){NA}),
            df = tryCatch({(t.test(percent_correct, mu=0.5)$parameter)},error = function(err){NA}),
            pval = tryCatch({(t.test(percent_correct, mu=0.5)$p.value)},error = function(err){NA}))%>%
  mutate(se = sd / sqrt(count)) %>%
  mutate(cohensd = (avgs - .5) / sd) %>%
  mutate(imp_agent = paste(imprint.cond, agent, sep="_"))

write.csv(by_test_cond, "stats_by_agent.csv")

for (i in unique(by_test_cond$imp_agent))
{
  bar_data <- by_test_cond %>%
    filter(imp_agent == i)

  img_name <- paste0(i, "_test.png")

  make_bar_charts(data = bar_data,
                  dots = NULL,
                  aes_y = avgs,
                  error_min = avgs - se,
                  error_max = avgs + se,
                  img_name = img_name)
}


# Plot by imprinting condition -------------------------------------------------
## Remove rest data once we start to group agents (for ease of presentation)

by_imp_cond <- by_test_cond %>%
  ungroup() %>%
  group_by(imprint.cond, test.cond) %>%
  summarise(avgs_by_imp = mean(avgs, na.rm = TRUE), 
            sd = sd(avgs, na.rm = TRUE), 
            count = length(avgs),
            tval = tryCatch({ (t.test(avgs, mu=0.5)$statistic)}, error = function(err){NA}),
            df = tryCatch({ (t.test(avgs, mu=0.5)$parameter)}, error = function(err){NA}),
            pval = tryCatch({ (t.test(avgs, mu=0.5)$p.value)}, error = function(err){NA}))%>%
  mutate(se = sd / sqrt(count)) %>%
  mutate(cohensd = (avgs_by_imp - .5) / sd)

write.csv(by_imp_cond, "stats_by_imp_cond.csv")

for (i in unique(by_imp_cond$imprint.cond))
{
  bar_data <- by_imp_cond %>%
    filter(imprint.cond == i) %>%
    filter(test.cond != "Rest")
  
  dot_data <- by_test_cond %>%
    filter(imprint.cond == i) %>%
    filter(test.cond != "Rest")

  img_name <- paste0(i, "_test.png")

  make_bar_charts(data = bar_data,
                  dots = dot_data,
                  aes_y = avgs_by_imp,
                  error_min = avgs_by_imp - se,
                  error_max = avgs_by_imp + se,
                  img_name = img_name)
}


# Plot across all imprinting conditions ----------------------------------------
across_imp_cond <- by_test_cond %>%
  ungroup() %>%
  filter(test.cond != "Rest") %>%
  group_by(test.cond) %>%
  summarise(all_avgs = mean(avgs, na.rm = TRUE), 
            sd = sd(avgs, na.rm = TRUE), 
            count = length(avgs),
            tval = tryCatch({ (t.test(avgs, mu=0.5)$statistic)}, error = function(err){NA}),
            df =  tryCatch({ (t.test(avgs, mu=0.5)$parameter)}, error = function(err){NA}),
            pval = tryCatch({ (t.test(avgs, mu=0.5)$p.value)}, error = function(err){NA}))%>%
  mutate(se = sd / sqrt(count)) %>%
  mutate(cohensd = (all_avgs - .5) / sd)

write.csv(across_imp_cond, "stats_across_all_agents.csv")

dot_data <- filter(by_test_cond, test.cond != "Rest")

make_bar_charts(data = across_imp_cond,
                dots = dot_data,
                aes_y = all_avgs,
                error_min = all_avgs - se,
                error_max = all_avgs + se,
                img_name = "all_imprinting_conds_test.png")
