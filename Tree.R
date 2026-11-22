library(readxl)
library(rpart)
library(rpart.plot)

df <- as.data.frame(read_excel("new_data.xlsx",
                               sheet = "model_data"))

# Target as a factor. "Threatened" first so "Endangered" is the positive class.
df$status <- factor(ifelse(df$is_endangered, "Endangered", "Threatened"),
                    levels = c("Threatened", "Endangered"))
green  <- c("has_recovery_doc", "has_five_year_review", "has_critical_habitat",
            "rec_n_documents", "rec_plan_stage", "fyr_review_in_progress",
            "ch_type")
yellow <- c("rec_has_final_plan", "fyr_recommendation")
added  <- c("species_group", "fws_lead_region", "listing_year", "is_plant",
            "is_population_segment", "ch_total_acres", "ch_total_miles",
            "rec_plan_revision")

cat_cols  <- c("rec_plan_stage", "ch_type", "fyr_recommendation",
               "species_group", "fws_lead_region")
bool_cols <- c("has_recovery_doc", "has_five_year_review", "has_critical_habitat",
               "fyr_review_in_progress", "rec_has_final_plan", "is_plant",
               "is_population_segment")
for (v in cat_cols)  df[[v]] <- factor(df[[v]])
levels(df$fws_lead_region) <- sub("Region ", "R", levels(df$fws_lead_region))
levels(df$fws_lead_region)[levels(df$fws_lead_region) == "No FWS region"] <- "None"
for (v in bool_cols) df[[v]] <- factor(ifelse(df[[v]] == 1, "Yes", "No"), levels = c("No", "Yes"))

# Train/test split by species (all rows of a species stay on one side)
species      <- unique(df$scientific_name)
test_species <- sample(species, round(0.30 * length(species)))
is_test      <- df$scientific_name %in% test_species
train <- df[!is_test, ]
test  <- df[ is_test, ]
cat(sprintf("Train: %d rows | Test: %d rows | %% Endangered (train): %.1f%%\n",
            nrow(train), nrow(test), 100 * mean(train$status == "Endangered")))

# 10 cross-validation folds inside the training set, also grouped by species.
# rpart accepts a vector of fold IDs as xval.
train_species <- unique(train$scientific_name)
fold_of <- setNames(sample(rep(1:10, length.out = length(train_species))), train_species)
cv_folds <- unname(fold_of[train$scientific_name])

# ---- Fit the tree with all features ----------------------------------------
f <- as.formula(paste("status ~", paste(c(green, yellow, added), collapse = " + ")))
full_tree <- rpart(f, data = train, method = "class",
                   parms   = list(split = "gini", prior = c(0.5, 0.5)),
                   control = rpart.control(cp = 0.001, minsplit = 20, minbucket = 7,
                                           maxdepth = 10, xval = cv_folds))
printcp(full_tree)
plotcp(full_tree)

# ---- Prune with the 1-SE rule ----------------------------------------------
ct    <- full_tree$cptable
best  <- which.min(ct[, "xerror"])
limit <- ct[best, "xerror"] + ct[best, "xstd"]
pick  <- which(ct[, "xerror"] <= limit)[1]
tree  <- prune(full_tree, cp = ct[pick, "CP"])

# ---- Plot (shows in RStudio's Plots pane; no file is saved) ----------------
wrap_labels <- function(x, labs, digits, varlen, faclen) {
  sapply(labs, function(l) paste(strwrap(gsub(",", ", ", l), width = 30), collapse = "\n"))
}

rpart.plot(tree, type = 2, extra = 106, under = TRUE, fallen.leaves = TRUE,
           box.palette = "BuRd", roundint = FALSE,
           split.fun = wrap_labels, split.cex = 0.9, tweak = 1.15,
           main = "Endangered vs Threatened - all features")

# ---- Evaluate on the test set ----------------------------------------------
p    <- predict(tree, test, type = "prob")[, "Endangered"]
pred <- factor(ifelse(p >= 0.5, "Endangered", "Threatened"),
               levels = c("Threatened", "Endangered"))
cm   <- table(Predicted = pred, Actual = test$status)
print(cm)

y    <- test$status == "Endangered"
r    <- rank(p)
auc  <- (sum(r[y]) - sum(y) * (sum(y) + 1) / 2) / (sum(y) * sum(!y))
sens <- cm["Endangered", "Endangered"] / sum(cm[, "Endangered"])
spec <- cm["Threatened", "Threatened"] / sum(cm[, "Threatened"])
cat(sprintf("Accuracy %.3f | Balanced accuracy %.3f | Sensitivity %.3f | Specificity %.3f | AUC %.3f | Leaves %d\n",
            mean(pred == test$status), (sens + spec) / 2, sens, spec, auc,
            sum(tree$frame$var == "<leaf>")))
cat(sprintf("Baseline (always guess Endangered): accuracy %.3f, AUC 0.5\n", mean(y)))
