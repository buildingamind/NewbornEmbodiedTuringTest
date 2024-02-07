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
parser$add_argument("--key-csv", type="character", dest="key_csv",
                    help="Full address of the key file that designates condition and correct monitor for each trial",
                    required=TRUE)
# parser$add_argument("--graphstyle", type="character", dest="graphstyle",
#                     help="Full address of the NETT_graphstyle.R file",
#                     required=TRUE)
parser$add_argument("--results-wd", type="character", dest="results_wd",
                    help="Working directory to save the resulting visualizations",
                    required=TRUE)
parser$add_argument("--color-bars", type = "character", dest="color_bars",
                    help="Should the bars be colored by test condition?",
                    required=TRUE)
args <- parser$parse_args()
data_loc <- args$data_loc; chick_file <- args$chick_file; key_csv <- args$key_csv
#graphstyle <- args$graphstyle
results_wd <- args$results_wd
if( args$color_bars %in% c("t", "T", "true", "TRUE")) {color_bars <- TRUE} else { color_bars <- FALSE}

# Set Up -----------------------------------------------------------------------

library(tidyverse)
library(stringr)

#source(graphstyle)

# Load the chick data
chick_data <- read.csv(chick_file)

# Load the condition key
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
test_data <- test_data %>%
  mutate(correct_steps = if_else(correct_monitor == "left", left_steps, right_steps)) %>%
  mutate(incorrect_steps = if_else(correct_monitor == "left", right_steps, left_steps)) %>%
  mutate(percent_correct = correct_steps / (correct_steps + incorrect_steps)) %>%
  mutate()

# Convert the levels of our bar variable into text so that line breaks show up
chick_data$cond_name <- gsub("LB", "\n", chick_data$cond_name)
test_data$cond_name <- gsub("LB", "\n", test_data$cond_name)

# Convert cond_name to factor so that it stays in the csv-specified order
original_order <- unique(chick_data$cond_name)
chick_data$cond_name <- factor(chick_data$cond_name, levels = original_order)

setwd(results_wd)


# Plot aesthetic settings ------------------------------------------------------
custom_palette <- c("#3F8CB7", "#FCEF88", "#5D5797", "#62AC6B", "#B74779")
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
                  aes(x=cond_name, ymin=avg, ymax=avg)) +

    # Model performance: bars
    {if(color_bars)geom_col(data = data, width = 0.7, aes(x=cond_name, y = {{aes_y}}, fill = cond_name))}+
    {if(!color_bars)geom_col(data = data, width = 0.7, aes(x=cond_name, y = {{aes_y}}), fill = "gray45")}+
    # Model performance: error bars
    geom_errorbar(data = data, width = 0.3,
                  aes(x = cond_name, ymin = {{error_min}}, ymax = {{error_max}})) +
    # Model performance: dots
    {if(!is.null(dots))geom_jitter(data = dot_data, aes(x=cond_name, y = avgs), width = .3)}+
    theme(legend.position="none") +

    # Add chicken performance again so that it shows up on top
    # Chick performance: lines (errorbar) with ribbons (crossbar)
    geom_errorbar(data=chick_data, width = 0.7, colour = chickred,
                aes(x=cond_name, ymin=avg, ymax=avg)) +
    geom_crossbar(data=chick_data, width = 0.7,
                  linetype = 0, fill = chickred, alpha = 0.2,
                  aes(x = cond_name, y = avg,
                      ymin = avg - avg_dev, ymax = avg + avg_dev))

  ggsave(img_name, width = 6, height = 6)
}


# Plot by agent ----------------------------------------------------------------
## Leave rest data for agent-level graphs

## Group data by test conditions
by_test_cond <- test_data %>%
  group_by(imprinting, agent, cond_name) %>%
  summarise(avgs = mean(percent_correct, na.rm = TRUE),
            sd = sd(percent_correct, na.rm = TRUE),
            count = length(percent_correct),
            tval = tryCatch({ (t.test(percent_correct, mu=0.5)$statistic)}, error = function(err){NA}),
            df = tryCatch({(t.test(percent_correct, mu=0.5)$parameter)},error = function(err){NA}),
            pval = tryCatch({(t.test(percent_correct, mu=0.5)$p.value)},error = function(err){NA}))%>%
  mutate(se = sd / sqrt(count)) %>%
  mutate(cohensd = (avgs - .5) / sd) %>%
  mutate(imp_agent = paste(imprinting, agent, sep="_"))

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
  group_by(imprinting, cond_name) %>%
  summarise(avgs_by_imp = mean(avgs, na.rm = TRUE),
            sd = sd(avgs, na.rm = TRUE),
            count = length(avgs),
            tval = tryCatch({ (t.test(avgs, mu=0.5)$statistic)}, error = function(err){NA}),
            df = tryCatch({ (t.test(avgs, mu=0.5)$parameter)}, error = function(err){NA}),
            pval = tryCatch({ (t.test(avgs, mu=0.5)$p.value)}, error = function(err){NA}))%>%
  mutate(se = sd / sqrt(count)) %>%
  mutate(cohensd = (avgs_by_imp - .5) / sd)

write.csv(by_imp_cond, "stats_by_imp_cond.csv")

for (i in unique(by_imp_cond$imprinting))
{
  bar_data <- by_imp_cond %>%
    filter(imprinting == i) %>%
    filter(cond_name != "Rest")

  dot_data <- by_test_cond %>%
    filter(imprinting == i) %>%
    filter(cond_name != "Rest")

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
  filter(cond_name != "Rest") %>%
  group_by(cond_name) %>%
  summarise(all_avgs = mean(avgs, na.rm = TRUE),
            sd = sd(avgs, na.rm = TRUE),
            count = length(avgs),
            tval = tryCatch({ (t.test(avgs, mu=0.5)$statistic)}, error = function(err){NA}),
            df =  tryCatch({ (t.test(avgs, mu=0.5)$parameter)}, error = function(err){NA}),
            pval = tryCatch({ (t.test(avgs, mu=0.5)$p.value)}, error = function(err){NA}))%>%
  mutate(se = sd / sqrt(count)) %>%
  mutate(cohensd = (all_avgs - .5) / sd)

write.csv(across_imp_cond, "stats_across_all_agents.csv")

dot_data <- filter(by_test_cond, cond_name != "Rest")

make_bar_charts(data = across_imp_cond,
                dots = dot_data,
                aes_y = all_avgs,
                error_min = all_avgs - se,
                error_max = all_avgs + se,
                img_name = "all_imprinting_conds_test.png")
