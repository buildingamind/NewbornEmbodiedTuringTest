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
parser$add_argument("--key-csv", type="character", dest="key_csv",
                    help="Full address of the key file that designates condition and correct monitor for each trial",
                    required=TRUE)
parser$add_argument("--results-wd", type="character", dest="results_wd",
                    help="Working directory to save the resulting visualizations",
                    required=TRUE)
parser$add_argument("--color-dots", type = "character", dest="color_dots",
                    help="For the graph with all agents combined, 
                    should the dots for each agent be colored by imprinting condition?",
                    required=TRUE)
args <- parser$parse_args()
data_loc <- args$data_loc; key_csv <- args$key_csv; results_wd <- args$results_wd
if( args$color_dots %in% c("t", "T", "true", "TRUE")) {color_dots = TRUE} else { color_dots = FALSE}



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
test_data <- test_data %>%
  mutate(correct_steps = if_else(correct_monitor == "left", left_steps, right_steps)) %>%
  mutate(incorrect_steps = if_else(correct_monitor == "left", right_steps, left_steps)) %>%
  mutate(percent_correct = correct_steps / (correct_steps + incorrect_steps)) %>%
  mutate()

setwd(results_wd)


# Plot aesthetic settings ------------------------------------------------------
p <- ggplot() + 
  theme_classic() + 
  theme(axis.text.x = element_text(size = 6)) + 
  ylab("Percent Correct") + 
  xlab("Test Condition") +
  scale_y_continuous(expand = c(0, 0), limits = c(0, 1), breaks=seq(0,1,.1), labels = scales::percent) +
  scale_x_discrete(labels = label_wrap(25)) +
  geom_hline(yintercept = .5, linetype = 2)


# Plot by agent ----------------------------------------------------------------
## Leave rest data for agent-level graphs

## Group data by test conditions
by_test_cond <- test_data %>%
  group_by(imprinting, agent, cond_name) %>%
  summarise(avgs = mean(percent_correct, na.rm = TRUE), 
            sd = sd(percent_correct, na.rm = TRUE), 
            count = length(percent_correct),
            tval =  ifelse(is.na(avgs),0.0,t.test(percent_correct, mu=0.5)$statistic),
            df =  ifelse(is.na(avgs),0.0,t.test(percent_correct, mu=0.5)$parameter),
            pval = ifelse(is.na(avgs),0.0,t.test(percent_correct, mu=0.5)$p.value))%>%
  mutate(se = sd / sqrt(count)) %>%
  mutate(cohensd = (avgs - .5) / sd) %>%
  mutate(imp_agent = paste(imprinting, agent, sep="_"))

write.csv(by_test_cond, "stats_by_agent.csv")

for (i in unique(by_test_cond$imp_agent))
{
  bar_data <- by_test_cond %>%
    filter(imp_agent == i)
  
  p + 
    geom_col(data = bar_data, width = 0.7, aes(x=cond_name, y = avgs)) + 
    geom_errorbar(data = bar_data, width = 0.3,
                  aes(x = cond_name, 
                      ymin = avgs - se, 
                      ymax = avgs + se))
    
  img_name <- paste0(i, "_test.png")
  ggsave(img_name)
}


# Plot by imprinting condition -------------------------------------------------
## Remove rest data once we start to group agents (for ease of presentation)

by_imp_cond <- by_test_cond %>%
  ungroup() %>%
  group_by(imprinting, cond_name) %>%
  summarise(avgs_by_imp = mean(avgs, na.rm = TRUE), 
            sd = sd(avgs, na.rm = TRUE), 
            count = length(avgs),
            tval = ifelse(is.na(avgs),0.0,t.test(avgs, mu=0.5)$statistic), 
            df = ifelse(is.na(avgs),0.0,t.test(avgs, mu=0.5)$parameter), 
            pval = ifelse(is.na(avgs),0.0,t.test(avgs, mu=0.5)$p.value))%>%
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
  
  p + 
    geom_col(data = bar_data, width = 0.7, aes(x=cond_name, y = avgs_by_imp)) + 
    geom_errorbar(data = bar_data, width = 0.3,
                  aes(x = cond_name, 
                      ymin = avgs_by_imp - se, 
                      ymax = avgs_by_imp + se)) +
    geom_jitter(data = dot_data, aes(x=cond_name, y = avgs), width = .3)
    
  
  img_name <- paste0(i, "_test.png")
  ggsave(img_name)
}


# Plot across all imprinting conditions ----------------------------------------
across_imp_cond <- by_test_cond %>%
  ungroup() %>%
  filter(cond_name != "Rest") %>%
  group_by(cond_name) %>%
  summarise(all_avgs = mean(avgs, na.rm = TRUE), 
            sd = sd(avgs, na.rm = TRUE), 
            count = length(avgs),
            tval =  ifelse(is.na(avgs),0.0,t.test(avgs, mu=0.5)$statistic), 
            df = ifelse(is.na(avgs),0.0,t.test(avgs, mu=0.5)$parameter), 
            pval = ifelse(is.na(avgs),0.0,t.test(avgs, mu=0.5)$p.value)) %>%
  mutate(se = sd / sqrt(count)) %>%
  mutate(cohensd = (all_avgs - .5) / sd)

write.csv(across_imp_cond, "stats_across_all_agents.csv")

p + 
  geom_col(data = across_imp_cond, width = 0.7, aes(x=cond_name, y = all_avgs)) + 
  geom_errorbar(data = across_imp_cond, width = 0.3,
                aes(x = cond_name, ymin = all_avgs - se, ymax = all_avgs + se)) +
  if(color_dots) {
    geom_jitter(data = filter(by_test_cond, cond_name != "Rest"), 
                aes(x = cond_name, y = avgs, colour = imprinting), width = .3)
  } else {
    geom_jitter(data = filter(by_test_cond, cond_name != "Rest"), 
                aes(x = cond_name, y = avgs), width = .3)
  }

ggsave("all_imprinting_conds_test.png")