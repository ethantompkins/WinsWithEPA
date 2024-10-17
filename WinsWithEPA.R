# Load necessary libraries
library(tidyverse)
library(ggplot2)
library(ggimage)
library(moments)
library(tidymodels)
library(nflfastR)
library(randomForest)

# Load play-by-play data from nflfastR
pbp <- load_pbp(1999:2023)

# Create outcomes dataframe
outcomes <- pbp %>%
  filter(week <= 17) %>% 
  group_by(season, game_id, home_team) %>%
  summarize(
    home_win = if_else(sum(result, na.rm = TRUE) > 0, 1, 0),
    home_tie = if_else(sum(result, na.rm = TRUE) == 0, 1, 0),
    home_diff = last(result, default = NA),
    home_pts_for = last(home_score, default = NA),
    home_pts_against = last(away_score, default = NA)
  ) %>%
  group_by(season, home_team) %>%
  summarize(
    home_games = n(),
    home_wins = sum(home_win),
    home_ties = sum(home_tie),
    home_diff = sum(home_diff, na.rm = TRUE),
    home_pts_for = sum(home_pts_for, na.rm = TRUE),
    home_pts_against = sum(home_pts_against, na.rm = TRUE)
  ) %>%
  ungroup() %>%
  left_join(
    pbp %>%
      filter(week <= 17) %>%
      group_by(season, game_id, away_team) %>%
      summarize(
        away_win = if_else(sum(result, na.rm = TRUE) < 0, 1, 0),
        away_tie = if_else(sum(result, na.rm = TRUE) == 0, 1, 0),
        away_diff = last(result, default = NA) * -1,
        away_pts_for = last(away_score, default = NA),
        away_pts_against = last(home_score, default = NA)
      ) %>%
      group_by(season, away_team) %>%
      summarize(
        away_games = n(),
        away_wins = sum(away_win),
        away_ties = sum(away_tie),
        away_diff = sum(away_diff, na.rm = TRUE),
        away_pts_for = sum(away_pts_for, na.rm = TRUE),
        away_pts_against = sum(away_pts_against, na.rm = TRUE)
      ) %>%
      ungroup(),
    by = c("season", "home_team" = "away_team")
  ) %>%
  rename(team = "home_team") %>%
  mutate(
    games = home_games + away_games,
    wins = home_wins + away_wins,
    losses = games - wins,
    ties = home_ties + away_ties,
    win_percentage = (wins + 0.5 * ties) / games,
    point_diff = home_diff + away_diff,
    points_for = home_pts_for + away_pts_for,
    points_against = home_pts_against + away_pts_against,
    pythag_wins = (points_for^2.37 / (points_for^2.37 + points_against^2.37)) * 16
  ) %>%
  select(
    season, team, games, wins, losses, ties, win_percentage, point_diff, points_for, points_against, pythag_wins
  )

# Create metrics dataframe
metrics <- pbp %>% 
  filter(
    week <= 17 & (pass == 1 | rush == 1) & !is.na(epa)
  ) %>% 
  group_by(season, posteam) %>% 
  summarize(
    n_pass = sum(pass, na.rm = TRUE),
    n_rush = sum(rush, na.rm = TRUE),
    pass_yards = sum(yards_gained * pass, na.rm = TRUE),
    rush_yards = sum(yards_gained * rush, na.rm = TRUE),
    epa_per_pass = sum(epa * pass, na.rm = TRUE) / n_pass,
    epa_per_rush = sum(epa * rush, na.rm = TRUE) / n_rush,
    success_per_pass = sum(pass * (epa > 0), na.rm = TRUE) / n_pass,
    success_per_rush = sum(rush * (epa > 0), na.rm = TRUE) / n_rush,
    y_per_pass = sum(yards_gained * pass, na.rm = TRUE) / n_pass,
    y_per_rush = sum(yards_gained * rush, na.rm = TRUE) / n_rush
  ) %>% 
  left_join(
    pbp %>%
      filter(
        week <= 17 & (pass == 1 | rush == 1) & !is.na(epa)
      ) %>% 
      group_by(season, defteam) %>% 
      summarize(
        def_n_pass = sum(pass, na.rm = TRUE),
        def_n_rush = sum(rush, na.rm = TRUE),
        def_pass_yards = sum(yards_gained * pass, na.rm = TRUE),
        def_rush_yards = sum(yards_gained * rush, na.rm = TRUE),
        def_epa_per_pass = sum(-epa * pass, na.rm = TRUE) / def_n_pass,
        def_epa_per_rush = sum(-epa * rush, na.rm = TRUE) / def_n_rush,
        def_success_per_pass = sum(pass * (epa > 0), na.rm = TRUE) / def_n_pass,
        def_success_per_rush = sum(rush * (epa > 0), na.rm = TRUE) / def_n_rush,
        def_y_per_pass = sum(yards_gained * pass, na.rm = TRUE) / def_n_pass,
        def_y_per_rush = sum(yards_gained * rush, na.rm = TRUE) / def_n_rush
      ),
    by = c("season", "posteam" = "defteam")
  ) %>% 
  rename(team = "posteam") %>% 
  select(-n_pass, -n_rush, -def_n_pass, -def_n_rush)

# Create dataframe for season-long outcomes and stats
df <- outcomes %>% 
  left_join(metrics, by = c("season", "team")) %>%
  drop_na() # Ensure no missing values

# Simple Linear Regression
r_squareds <- c()
for (i in 12:ncol(df)) {
  input = colnames(df)[i]
  fit <- lm(data = df, wins ~ get(input))
  r2 <- summary(fit)$r.squared
  r_squareds = rbind(r_squareds, data.frame(input, r2))
}

# Random Forest Variable Importance
set.seed(12)
crs <- list()
crs$dataset <- df
crs$train <- sample(nrow(crs$dataset), 0.7 * nrow(crs$dataset))
crs$input <- c(
  "pass_yards", "rush_yards", "epa_per_pass", "epa_per_rush", "success_per_pass",
  "success_per_rush", "y_per_pass", "y_per_rush", "def_pass_yards", "def_rush_yards",
  "def_epa_per_pass", "def_epa_per_rush", "def_success_per_pass", "def_success_per_rush",
  "def_y_per_pass", "def_y_per_rush"
)
crs$target <- "wins"
crs$rf <- randomForest::randomForest(
  wins ~ ., data = crs$dataset[crs$train, c(crs$input, crs$target)],
  ntree = 500, mtry = 4, importance = TRUE, na.action = randomForest::na.roughfix, replace = FALSE
)
rf_imp <- as.data.frame(crs$rf$importance)

# Multiple Linear Regression
fit <- lm(data = df, wins ~ epa_per_pass + epa_per_rush + def_epa_per_pass + def_epa_per_rush)
df$pred <- predict(fit, type = "response")
df$var <- df$wins - df$pred

# Plotting for Expected vs Actual Wins for 2023
plot_function <- function(df, szn) {
  df <- df %>% filter(season == szn) %>% arrange(-var)
  df$team <- factor(df$team, levels = df$team)
  ggplot(df, aes(x = reorder(team, var), y = var)) +
    geom_bar(stat = "identity", aes(color = team, fill = team), show.legend = FALSE) +
    labs(
      title = paste(szn, "Actual Wins over Expected Wins"),
      subtitle = "Expected wins based on season EPA metrics",
      x = element_blank(), y = element_blank(), caption = "Data from @nflscrapR & @nflfastR"
    ) +
    coord_flip()
}

# Example usage of plot function for 2023
plot_function(df, 2023)
