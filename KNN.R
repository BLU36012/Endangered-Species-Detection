# k-nearest neighbors (Amanda)
#
# Mirrors Tree.R so the two models compare fairly: same file, same target,
# same features, same species-grouped 70/30 split and 10 grouped CV folds.
# kNN has no class weights, so the minority class is oversampled inside the
# training folds only (never in the validation fold or the test set).
# Features are one-hot encoded and scaled with training-set means/SDs only.

library(readxl)
library(class)

set.seed(160)  

# read the data
df <- as.data.frame(read_excel("new_data.xlsx", sheet = "model_data"))

# Features for the model, same as in Tree.R
green  <- c("has_recovery_doc", "has_five_year_review", "has_critical_habitat",
            "rec_n_documents", "rec_plan_stage", "fyr_review_in_progress",
            "ch_type")
yellow <- c("rec_has_final_plan", "fyr_recommendation")
added  <- c("species_group", "fws_lead_region", "listing_year", "is_plant",
            "is_population_segment", "ch_total_acres", "ch_total_miles",
            "rec_plan_revision")
features <- c(green, yellow, added)

# Target: Endangered vs Threatened. "Threatened" first so "Endangered" is the
# positive class, as in Tree.R. Threatened (~22%) is the minority class.
# turns the binary indicator into a factor with levels "Threatened" and "Endangered"
df$status <- factor(ifelse(df$is_endangered == 1, "Endangered", "Threatened"),
                    levels = c("Threatened", "Endangered"))
# Define the positive and negative classes, and the minority class for evaluation
pos      <- "Endangered"
neg      <- "Threatened"
minority <- "Threatened"

# ---- Feature matrix ---------------------------------------------------------
# Acres and miles span 0 to 120 million, so log them before scaling or they
# swamp every other distance.
# log-transform acres and miles first
df$ch_total_acres <- log1p(df$ch_total_acres)
df$ch_total_miles <- log1p(df$ch_total_miles)

# turn all of these features into factors
cat_cols <- intersect(c("rec_plan_stage", "ch_type", "fyr_recommendation",
                        "species_group", "fws_lead_region"), features)
for (v in cat_cols) df[[v]] <- factor(df[[v]])
# one-hot encode all features and create a feature matrix
X <- model.matrix(~ has_recovery_doc + has_five_year_review + has_critical_habitat +
                    rec_n_documents + rec_plan_stage + fyr_review_in_progress +
                    ch_type + rec_has_final_plan + fyr_recommendation +
                    species_group + fws_lead_region + listing_year + is_plant +
                    is_population_segment + ch_total_acres + ch_total_miles +
                    rec_plan_revision,
                  data = df)
# drop the intercept column from the model matrix
X <- X[, -1]                                  
# ---- Split by species (all rows of a species stay on one side) --------------
species      <- unique(df$scientific_name)
# randomly sample 30% of species for the test set
test_species <- sample(species, round(0.30 * length(species)))
# create a logical vector indicating which rows belong to the test set
is_test      <- df$scientific_name %in% test_species
# create training and test sets based on the logical vector
Xtr <- X[!is_test, ]
# create the corresponding target variable for the training set
ytr <- df$status[!is_test]
Xte <- X[ is_test, ]
yte <- df$status[ is_test]
cat(sprintf("Train: %d rows | Test: %d rows | %% Endangered (train): %.1f%%\n",
            nrow(Xtr), nrow(Xte), 100 * mean(ytr == pos)))

# --- 10 cross-validation folds inside the training set, also grouped by species.
train_species <- unique(df$scientific_name[!is_test])
fold_of  <- setNames(sample(rep(1:10, length.out = length(train_species))), train_species)
cv_folds <- unname(fold_of[df$scientific_name[!is_test]])

# ---- Helpers -----------------------------------------------------------------
# Oversample the minority class in the training folds only, never in the validation fold or test set.
oversample <- function(X, y) {
  # size of the larger class
  n_max <- max(table(y))
  # for each class, find the indices of the rows belonging to that class, and sample with replacement to create a balanced dataset
  idx <- unlist(lapply(levels(y), function(l) {
    # find the indices of the rows belonging to class l
    i <- which(y == l)
    # sample with replacement to create a balanced dataset
    c(i, sample(i, n_max - length(i), replace = TRUE))
  }))
  # return a list containing the oversampled feature matrix and target variable
  list(X = X[idx, , drop = FALSE], y = y[idx])
}

# Scale a feature matrix using the provided means and standard deviations, handling zero SDs to avoid division by zero.
scale_by <- function(X, center, sd) {
  # replace any zero standard deviations with 1 to avoid division by zero
  sd[sd == 0] <- 1
  # scale the feature matrix by subtracting the means and dividing by the standard deviations
  sweep(sweep(X, 2, center, "-"), 2, sd, "/")
}

# Fit a kNN model and make predictions
fit_predict <- function(Xa, ya, Xb, k) {
  # compute the means of the training set for scaling
  mu <- colMeans(Xa)
  # compute the standard deviations of the training set for scaling
  s <- apply(Xa, 2, sd)  
  # oversample the training set to balance the classes
  bal <- oversample(Xa, ya)                    
  # make predictions on the test set using the kNN algorithm, scaling both the training and test sets using the training set's means and standard deviations
  pr  <- knn(scale_by(bal$X, mu, s), scale_by(Xb, mu, s), bal$y, k = k, prob = TRUE)
  # compute the predicted probabilities for the positive class, using the attributes returned by the knn function
  p_pos <- ifelse(pr == pos, attr(pr, "prob"), 1 - attr(pr, "prob"))
  # return a list containing the predicted classes and predicted probabilities for the positive class
  list(pred = factor(pr, levels = levels(ya)), p = p_pos)
}

# scores predictions using precision, recall, and F1 score for a given class
metrics <- function(pred, actual, cls) {
  # compute the number of true positives for the given class
  tp <- sum(pred == cls & actual == cls)
  # compute precision, recall, and F1 score for the given class, handling cases where the denominator is zero to avoid division by zero errors
  prec <- tp / max(sum(pred == cls), 1)
  rec  <- tp / max(sum(actual == cls), 1)
  f1   <- ifelse(prec + rec == 0, 0, 2 * prec * rec / (prec + rec))
  c(precision = prec, recall = rec, F1 = f1)
}

# ---- Tune k with grouped 10-fold CV ------------------------------------------
# Picked on F1 for the minority class, per the ground rules.
ks <- seq(1, 51, by = 2)
# Run grouped 10-fold CV for each k, oversampling the training folds only, and compute metrics for the minority class.
cv <- t(sapply(ks, function(k) {
  # for each fold, make predictions on the validation fold using the training folds, and compute metrics for the minority class
  pred <- factor(rep(NA, length(ytr)), levels = levels(ytr))
  # for each fold, make predictions on the validation fold using the training folds, and compute metrics for the minority class
  for (f in 1:10) {
    va <- cv_folds == f
    pred[va] <- fit_predict(Xtr[!va, ], ytr[!va], Xtr[va, ], k)$pred
  }
  # compute metrics for the minority class, as well as balanced accuracy
  m <- metrics(pred, ytr, minority)
  # compute balanced accuracy as the average of the recall for the positive and negative classes
  bal_acc <- mean(c(metrics(pred, ytr, pos)["recall"], metrics(pred, ytr, neg)["recall"]))
  # return a vector containing the current value of k, the metrics for the minority class, and the balanced accuracy
  c(k = k, m, balanced_acc = bal_acc)
}))
print(round(cv, 3))
# Pick the best k based on the maximum F1 score for the minority class, and print the result.
best_k <- cv[which.max(cv[, "F1"]), "k"]
cat(sprintf("Best k by CV F1 (%s): %d\n", minority, best_k))

# Plot the cross-validated F1 scores against k
dir.create("figures", showWarnings = FALSE)
png("figures/knn_cv_k.png", width = 900, height = 600, res = 130)
plot(cv[, "k"], cv[, "F1"], type = "b", pch = 19, xlab = "k (neighbors)",
     ylab = paste("Cross-validated F1,", minority),
     main = "kNN: choosing k")
abline(v = best_k, lty = 2, col = "red")
dev.off()

# ---- Evaluate on the test set ------------------------------------------------
# Fit the kNN model on the entire training set with the best k, and make predictions on the test set. Then compute and print the confusion matrix, AUC, sensitivity, specificity, accuracy, balanced accuracy, and F1 score for each class.
out  <- fit_predict(Xtr, ytr, Xte, best_k)
pred <- out$pred; p <- out$p
cm   <- table(Predicted = pred, Actual = yte)
print(cm)

# Compute AUC, sensitivity, specificity, accuracy, balanced accuracy, and F1 score for each class
y    <- yte == pos
r    <- rank(p)
auc  <- (sum(r[y]) - sum(y) * (sum(y) + 1) / 2) / (sum(y) * sum(!y))
sens <- cm[pos, pos] / sum(cm[, pos])
spec <- cm[neg, neg] / sum(cm[, neg])
cat(sprintf("Accuracy %.3f | Balanced accuracy %.3f | Sensitivity %.3f | Specificity %.3f | AUC %.3f | k %d\n",
            mean(pred == yte), (sens + spec) / 2, sens, spec, auc, best_k))
for (cls in levels(yte)) {
  m <- metrics(pred, yte, cls)
  cat(sprintf("%-15s precision %.3f | recall %.3f | F1 %.3f\n", cls, m[1], m[2], m[3]))
}
cat(sprintf("Baseline (always guess Endangered): accuracy %.3f, Threatened recall 0\n",
            mean(yte == pos)))
