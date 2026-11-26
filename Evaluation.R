# Shared evaluation code (Krista)
# Everyone can use this with: source("Evaluation.R")

library(readxl)

# The columns we use to make guesses
FEATURES <- c("has_recovery_doc", "has_five_year_review", "has_critical_habitat",
              "rec_n_documents", "rec_plan_stage", "fyr_review_in_progress",
              "ch_type", "rec_has_final_plan", "fyr_recommendation",
              "species_group", "fws_lead_region", "listing_year", "is_plant",
              "is_population_segment", "ch_total_acres", "ch_total_miles",
              "rec_plan_revision")

# These columns are words, not numbers
WORD_COLUMNS <- c("rec_plan_stage", "ch_type", "fyr_recommendation",
                  "species_group", "fws_lead_region")

POS      <- "Endangered"
NEG      <- "Threatened"
MINORITY <- "Threatened"   # the rare group. We care most about finding these.


# Read the data
load_data <- function(path = "new_data.xlsx") {
  df <- as.data.frame(read_excel(path, sheet = "model_data"))

  # Make the answer column: "Endangered" or "Threatened"
  df$status <- ifelse(df$is_endangered == 1, "Endangered", "Threatened")
  df$status <- factor(df$status, levels = c("Threatened", "Endangered"))

  # Turn word columns into categories
  for (col in WORD_COLUMNS) {
    df[[col]] <- factor(df[[col]])
  }
  return(df)
}


# Turn the data into a table of only numbers
design_matrix <- function(df) {
  # Acres and miles can be HUGE, so shrink them with log
  df$ch_total_acres <- log1p(df$ch_total_acres)
  df$ch_total_miles <- log1p(df$ch_total_miles)

  # Make a formula like: ~ has_recovery_doc + has_five_year_review + ...
  my_formula <- as.formula(paste("~", paste(FEATURES, collapse = " + ")))

  # Turn each word category into 0/1 columns
  X <- model.matrix(my_formula, data = df)

  # Remove the first column (it is just all 1s)
  X <- X[, -1]
  return(X)
}


# Split into train and test. A species is never in both.
make_split <- function(df, seed = 160) {
  set.seed(seed)

  # Pick 30% of species for the test
  species <- unique(df$scientific_name)
  test_species <- sample(species, round(0.30 * length(species)))
  is_test <- df$scientific_name %in% test_species

  # Give each training species a fold number from 1 to 10
  train_species <- unique(df$scientific_name[is_test == FALSE])
  fold_numbers <- rep(1:10, length.out = length(train_species))
  fold_numbers <- sample(fold_numbers)
  names(fold_numbers) <- train_species

  # Look up the fold for every training row
  folds <- fold_numbers[df$scientific_name[is_test == FALSE]]
  folds <- unname(folds)

  return(list(is_test = is_test, folds = folds, n_folds = 10))
}


# Give EVERY species (train and test) a fold number from 1 to 10
make_all_folds <- function(df, seed = 160) {
  set.seed(seed)
  species <- unique(df$scientific_name)
  fold_numbers <- rep(1:10, length.out = length(species))
  fold_numbers <- sample(fold_numbers)
  names(fold_numbers) <- species
  return(unname(fold_numbers[df$scientific_name]))
}


# Precision, recall and F1 for one group
class_metrics <- function(pred, actual, group) {
  # Right guesses for this group
  right <- sum(pred == group & actual == group)

  # Precision: when we guessed this group, how often were we right?
  precision <- right / max(sum(pred == group), 1)

  # Recall: of the real ones in this group, how many did we find?
  recall <- right / max(sum(actual == group), 1)

  # F1: one score that mixes precision and recall
  if (precision + recall == 0) {
    f1 <- 0
  } else {
    f1 <- 2 * precision * recall / (precision + recall)
  }
  return(c(precision = precision, recall = recall, F1 = f1))
}


# AUC: how well the scores rank Threatened above Endangered (0.5 = coin flip, 1 = perfect)
auc_minority <- function(probs, actual) {
  if (length(unique(probs)) == 1) {
    return(0.5)
  }
  is_thr <- actual == MINORITY
  n_thr <- sum(is_thr)
  n_end <- sum(!is_thr)
  ranks <- rank(probs)
  auc <- (sum(ranks[is_thr]) - n_thr * (n_thr + 1) / 2) / (n_thr * n_end)
  return(auc)
}


# All the scores for one model, in one row
score <- function(pred, actual, probs = NULL) {
  pred <- factor(pred, levels = levels(actual))
  thr <- class_metrics(pred, actual, "Threatened")
  end <- class_metrics(pred, actual, "Endangered")

  if (is.null(probs)) {
    auc <- NA
  } else {
    auc <- auc_minority(probs, actual)
  }

  return(c(accuracy      = mean(pred == actual),
           balanced_acc  = (thr[["recall"]] + end[["recall"]]) / 2,
           thr_precision = thr[["precision"]],
           thr_recall    = thr[["recall"]],
           thr_F1        = thr[["F1"]],
           end_F1        = end[["F1"]],
           AUC           = auc))
}


# Print the scores and the confusion matrix
evaluate <- function(pred, actual, probs = NULL, label = "") {
  pred <- factor(pred, levels = levels(actual))
  cat("\n====", label, "====\n")

  # Confusion matrix: guesses vs real answers
  print(table(Predicted = pred, Actual = actual))

  # Scores for each group
  for (group in levels(actual)) {
    m <- class_metrics(pred, actual, group)
    cat(sprintf("%-11s precision %.3f | recall %.3f | F1 %.3f\n",
                group, m[["precision"]], m[["recall"]], m[["F1"]]))
  }

  s <- score(pred, actual, probs)
  cat(sprintf("Accuracy %.3f | Balanced accuracy %.3f | AUC %.3f\n",
              s[["accuracy"]], s[["balanced_acc"]], s[["AUC"]]))
  return(invisible(s))
}
