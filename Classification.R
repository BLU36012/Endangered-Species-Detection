# My part (Krista): baseline, logistic regression, pick the best model, watchlist
#
# The tree and kNN parts copy Ryan's and Amanda's settings.
# They run here only so every model is tested on the same folds.
# This file does NOT change their files.

source("Evaluation.R")
library(rpart)
library(class)

dir.create("figures", showWarnings = FALSE)


# Get the data

df <- load_data()
X  <- design_matrix(df)

split      <- make_split(df)
train_rows <- which(split$is_test == FALSE)
test_rows  <- which(split$is_test == TRUE)
y_train    <- df$status[train_rows]
y_test     <- df$status[test_rows]
folds      <- split$folds

cat("Train rows:", length(train_rows), "| Test rows:", length(test_rows), "\n")


# Helper functions

# Make every column the same size, using the TRAINING averages
scale_like_train <- function(X_new, avg, spread) {
  spread[spread == 0] <- 1           # don't divide by 0
  return(scale(X_new, center = avg, scale = spread))
}

# Turn chances into guesses. Above the line = "Threatened".
make_guesses <- function(probs, line = 0.5) {
  guesses <- ifelse(probs >= line, "Threatened", "Endangered")
  return(factor(guesses, levels = c("Threatened", "Endangered")))
}


# The models
# Each one learns from rows "a" and gives a chance of "Threatened" for rows "b".

# Baseline: always guess Endangered (chance of Threatened = 0)
baseline_probs <- function(a, b) {
  return(rep(0, length(b)))
}

# Logistic regression
logreg_probs <- function(a, b, give_model = FALSE) {
  X_a <- X[a, ]
  X_b <- X[b, ]

  # Averages and spreads from the training rows
  avg    <- colMeans(X_a)
  spread <- apply(X_a, 2, sd)

  # Throw away columns that never change
  keep <- spread > 0
  train_data <- as.data.frame(scale_like_train(X_a[, keep], avg[keep], spread[keep]))
  new_data   <- as.data.frame(scale_like_train(X_b[, keep], avg[keep], spread[keep]))

  # 1 = Threatened, 0 = Endangered
  y <- ifelse(df$status[a] == "Threatened", 1, 0)

  # Threatened is rare, so each one counts more (balanced weights)
  n     <- length(y)
  n_thr <- sum(y == 1)
  n_end <- sum(y == 0)
  w <- ifelse(y == 1, n / (2 * n_thr), n / (2 * n_end))

  # Fit the model
  train_data$y <- y
  model <- suppressWarnings(glm(y ~ ., data = train_data, family = binomial, weights = w))

  # Chance of Threatened for the new rows
  probs <- suppressWarnings(predict(model, new_data, type = "response"))

  if (give_model) {
    return(model)
  }
  return(probs)
}

# Decision tree (Ryan's settings)
tree_formula <- as.formula(paste("status ~", paste(FEATURES, collapse = " + ")))

tree_probs <- function(a, b, give_model = FALSE) {
  # Grow a big tree
  big_tree <- rpart(tree_formula, data = df[a, ], method = "class",
                    parms   = list(split = "gini", prior = c(0.5, 0.5)),
                    control = rpart.control(cp = 0.001, minsplit = 20, minbucket = 7,
                                            maxdepth = 10, xval = 10))

  # Cut it back to a smaller tree (the "1-SE rule")
  cp_table   <- big_tree$cptable
  best       <- which.min(cp_table[, "xerror"])
  good_enough <- cp_table[best, "xerror"] + cp_table[best, "xstd"]
  pick       <- which(cp_table[, "xerror"] <= good_enough)[1]
  small_tree <- prune(big_tree, cp = cp_table[pick, "CP"])

  if (give_model) {
    return(small_tree)
  }
  probs <- predict(small_tree, df[b, ], type = "prob")[, "Threatened"]
  return(probs)
}

# k-nearest neighbors (Amanda's settings)
knn_probs <- function(a, b, k) {
  X_a <- X[a, ]
  y_a <- df$status[a]

  avg    <- colMeans(X_a)
  spread <- apply(X_a, 2, sd)

  # Copy Threatened rows until both groups are the same size (training rows only)
  thr_rows <- which(y_a == "Threatened")
  end_rows <- which(y_a == "Endangered")
  n_extra  <- length(end_rows) - length(thr_rows)
  extra    <- sample(thr_rows, n_extra, replace = TRUE)
  all_rows <- c(thr_rows, extra, end_rows)

  # Find the k closest training species for each new species
  guesses <- knn(scale_like_train(X_a[all_rows, ], avg, spread),
                 scale_like_train(X[b, ], avg, spread),
                 y_a[all_rows], k = k, prob = TRUE)

  # knn tells us how sure it is about its guess. Turn that into a chance of Threatened.
  sure  <- attr(guesses, "prob")
  probs <- ifelse(guesses == "Threatened", sure, 1 - sure)
  return(probs)
}

# Run any model by its name
get_probs <- function(model_name, a, b, k = 3) {
  if (model_name == "Baseline (always Endangered)") return(baseline_probs(a, b))
  if (model_name == "Logistic regression")          return(logreg_probs(a, b))
  if (model_name == "Decision tree")                return(tree_probs(a, b))
  if (model_name == "k-nearest neighbors")          return(knn_probs(a, b, k))
}


# Cross-validation
# Hide one fold, learn from the other 9, guess the hidden one. Do this 10 times.

cv_probs <- function(model_name, k = 3) {
  set.seed(160)
  probs <- rep(0, length(train_rows))
  for (f in 1:10) {
    hidden <- folds == f
    probs[hidden] <- get_probs(model_name, train_rows[!hidden], train_rows[hidden], k)
  }
  return(probs)
}


# Pick k for kNN

k_values <- seq(1, 51, by = 2)
f1_for_k <- c()
for (k in k_values) {
  probs    <- cv_probs("k-nearest neighbors", k)
  s        <- score(make_guesses(probs), y_train)
  f1_for_k <- c(f1_for_k, s[["thr_F1"]])
}
best_k <- k_values[which.max(f1_for_k)]
cat("Best k for kNN:", best_k, "\n")


# Compare all models

model_names <- c("Baseline (always Endangered)", "Logistic regression",
                 "Decision tree", "k-nearest neighbors")

all_cv_probs <- list()
cv_results   <- NULL
for (m in model_names) {
  probs <- cv_probs(m, best_k)
  all_cv_probs[[m]] <- probs
  cv_results <- rbind(cv_results, score(make_guesses(probs), y_train, probs))
}
rownames(cv_results) <- model_names

cat("\nCross-validation results (threshold 0.5)\n")
print(round(cv_results, 3))
write.csv(round(cv_results, 3), "figures/model_comparison_cv.csv")


# Pick the final model: best F1 for Threatened

final_name <- model_names[which.max(cv_results[, "thr_F1"])]
cat("\nFinal model:", final_name, "\n")

# Try different lines (thresholds) and keep the one with the best F1
lines  <- seq(0.05, 0.95, by = 0.01)
f1_for_line <- c()
for (line in lines) {
  s <- score(make_guesses(all_cv_probs[[final_name]], line), y_train)
  f1_for_line <- c(f1_for_line, s[["thr_F1"]])
}
best_line <- lines[which.max(f1_for_line)]
cat("Best threshold:", best_line, "| CV F1:", round(max(f1_for_line), 3), "\n")

png("figures/final_threshold.png", width = 900, height = 600, res = 130)
plot(lines, f1_for_line, type = "l", lwd = 2,
     xlab = "Threshold on P(Threatened)", ylab = "Cross-validated F1, Threatened",
     main = paste(final_name, ": choosing the threshold"))
abline(v = best_line, lty = 2, col = "red")
dev.off()


# Test set (used only once, at the end)

cat("\n######## Test set ########\n")
test_results <- NULL
for (m in model_names) {
  set.seed(160)
  probs <- get_probs(m, train_rows, test_rows, best_k)

  # The final model uses its best line. The others use 0.5.
  line <- 0.5
  if (m == final_name) {
    line <- best_line
  }
  s <- evaluate(make_guesses(probs, line), y_test, probs, paste(m, "- threshold", line))
  test_results <- rbind(test_results, s)
}
rownames(test_results) <- model_names
write.csv(round(test_results, 3), "figures/model_comparison_test.csv")

# Bar chart of CV F1
png("figures/model_comparison.png", width = 1000, height = 600, res = 130)
par(mar = c(5, 13, 3, 1))
barplot(rev(cv_results[, "thr_F1"]), horiz = TRUE, las = 1, xlim = c(0, 1),
        col = "steelblue", xlab = "Cross-validated F1, Threatened",
        main = "Model comparison")
dev.off()


# Top features (from logistic regression)
# Big + number = pushes toward Threatened. Big - number = pushes toward Endangered.

lr_model <- logreg_probs(train_rows, test_rows, give_model = TRUE)
coefs <- coef(lr_model)
coefs <- coefs[-1]                       # drop the intercept
coefs <- coefs[!is.na(coefs)]            # drop empty ones
coefs <- coefs[order(-abs(coefs))]       # biggest first
top   <- head(coefs, 15)

cat("\nTop features (logistic regression)\n")
print(round(top, 3))
write.csv(data.frame(feature = names(coefs), coef = round(coefs, 4)),
          "figures/logreg_coefficients.csv", row.names = FALSE)

png("figures/logreg_top_coefficients.png", width = 1000, height = 700, res = 130)
par(mar = c(5, 16, 3, 1))
bar_colors <- ifelse(rev(top) > 0, "darkorange", "steelblue")
barplot(rev(top), horiz = TRUE, las = 1, col = bar_colors,
        xlab = "Coefficient (scaled features)", main = "Logistic regression: top features")
abline(v = 0)
dev.off()

if (final_name == "Decision tree") {
  t <- tree_probs(train_rows, test_rows, give_model = TRUE)
  cat("\nDecision tree: most important features\n")
  print(round(head(t$variable.importance, 10), 1))
}


# Customer watchlist
# Give every species a score from a model that never saw it.
# Watchlist = Endangered species that LOOK like Threatened ones
#             AND are missing a recovery plan or critical habitat.

all_folds <- make_all_folds(df)
set.seed(160)
p_final  <- rep(0, nrow(df))
p_logreg <- rep(0, nrow(df))
for (f in 1:10) {
  hidden <- all_folds == f
  p_final[hidden]  <- get_probs(final_name, which(!hidden), which(hidden), best_k)
  p_logreg[hidden] <- logreg_probs(which(!hidden), which(hidden))   # second opinion
}
df$p_threatened        <- round(p_final, 3)
df$p_threatened_logreg <- round(p_logreg, 3)
df$both_models_agree   <- df$p_threatened_logreg >= 0.5

# Keep only the species we want
is_endangered <- df$status == "Endangered"
looks_thr     <- df$p_threatened >= best_line
no_plan       <- df$rec_has_final_plan == 0
no_habitat    <- df$has_critical_habitat == 0
watch <- df[is_endangered & looks_thr & (no_plan | no_habitat), ]

# Say what each species is missing
watch$missing <- ifelse(watch$rec_has_final_plan == 0 & watch$has_critical_habitat == 0,
                        "recovery plan + critical habitat",
                        ifelse(watch$rec_has_final_plan == 0,
                               "recovery plan", "critical habitat"))

# Sort: both models agree first, then highest score first
watch <- watch[order(-watch$both_models_agree, -watch$p_threatened, -watch$p_threatened_logreg), ]
watch <- watch[, c("scientific_name", "common_name", "species_group", "fws_lead_region",
                   "listing_year", "p_threatened", "p_threatened_logreg",
                   "both_models_agree", "missing", "fyr_recommendation")]

cat("\nWatchlist:", nrow(watch), "species |",
    sum(watch$both_models_agree), "where logistic regression agrees\n")
print(head(watch, 20), row.names = FALSE)
write.csv(watch, "figures/customer_watchlist.csv", row.names = FALSE)
