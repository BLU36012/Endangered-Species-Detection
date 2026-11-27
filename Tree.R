# Decision tree (Ryan)
source("Evaluation.R")
library(rpart)
library(rpart.plot)

#Can be changed depending whats needed
USE_FYR_RECOMMENDATION <- TRUE

# Data 
df <- load_data()   # adds df$status (Threatened / Endangered) and factors

# TRUE/FALSE (1/0) columns -> No/Yes, so the tree reads "has_recovery_doc = No"
bool_cols <- c("has_recovery_doc", "has_five_year_review", "has_critical_habitat",
               "fyr_review_in_progress", "rec_has_final_plan", "is_plant",
               "is_population_segment")
for (v in bool_cols) df[[v]] <- factor(ifelse(df[[v]] == 1, "Yes", "No"), levels = c("No", "Yes"))

# Shorter region names so the tree labels fit ("Region 1" -> "R1")
levels(df$fws_lead_region) <- sub("Region ", "R", levels(df$fws_lead_region))
levels(df$fws_lead_region)[levels(df$fws_lead_region) == "No FWS region"] <- "None"

features <- FEATURES
if (!USE_FYR_RECOMMENDATION) features <- setdiff(features, "fyr_recommendation")

#Shared train/test split and CV folds
split <- make_split(df)            # same split as Classification.R and KNN.R
train <- df[!split$is_test, ]
test  <- df[ split$is_test, ]
cv_folds <- split$folds            # one fold number per training row, grouped by species
cat(sprintf("Train: %d rows | Test: %d rows | %% Endangered (train): %.1f%%\n",
            nrow(train), nrow(test), 100 * mean(train$status == "Endangered")))

# Grow a large tree
f <- as.formula(paste("status ~", paste(features, collapse = " + ")))
full_tree <- rpart(f, data = train, method = "class",
                   parms   = list(split = "gini", prior = c(0.5, 0.5)),
                   control = rpart.control(cp = 0.001, minsplit = 20, minbucket = 7,
                                           maxdepth = 10, xval = cv_folds))
printcp(full_tree)
plotcp(full_tree)

#Prune with the 1-SE rule
ct    <- full_tree$cptable
best  <- which.min(ct[, "xerror"])
limit <- ct[best, "xerror"] + ct[best, "xstd"]
pick  <- which(ct[, "xerror"] <= limit)[1]
tree  <- prune(full_tree, cp = ct[pick, "CP"])

print(tree)   # text version of the tree

#Test set (same scores as Classification.R
p_thr <- predict(tree, test, type = "prob")[, "Threatened"]
pred  <- factor(ifelse(p_thr >= 0.5, "Threatened", "Endangered"),
                levels = c("Threatened", "Endangered"))
evaluate(pred, test$status, p_thr,
         paste("Decision tree", ifelse(USE_FYR_RECOMMENDATION, "", "(no fyr_recommendation)")))
cat(sprintf("Leaves: %d | Baseline (always Endangered) accuracy: %.3f\n",
            sum(tree$frame$var == "<leaf>"), mean(test$status == "Endangered")))

#Variable importance
vi <- tree$variable.importance
print(round(100 * vi / sum(vi), 1))
par(mar = c(4, 12, 2, 1))
barplot(rev(100 * vi / sum(vi)), horiz = TRUE, las = 1,
        xlab = "% of total importance", main = "Decision tree: variable importance")
par(mar = c(5, 4, 4, 2) + 0.1)   # reset margins

#Which species are in which leaf
train$leaf <- as.integer(rownames(tree$frame)[tree$where])
test$leaf  <- rpart.predict(tree, newdata = test, nn = TRUE)$nn

species_leaves <- rbind(
  data.frame(set = "train", train[, c("common_name", "scientific_name", "status", "leaf")],
             predicted = as.character(predict(tree, train, type = "class"))),
  data.frame(set = "test",  test[,  c("common_name", "scientific_name", "status", "leaf")],
             predicted = as.character(pred))
)
species_leaves$correct <- species_leaves$predicted == as.character(species_leaves$status)
species_leaves <- species_leaves[order(species_leaves$leaf, species_leaves$scientific_name), ]

print(rpart.rules(tree, nn = TRUE, cover = TRUE, roundint = FALSE))   # rule for each leaf
print(table(Leaf = species_leaves$leaf, Status = species_leaves$status))
View(species_leaves)
# One leaf, e.g. node 5:  subset(species_leaves, leaf == 5)

#Plot the tree (drawn last so it is the plot showing)
wrap_labels <- function(x, labs, digits, varlen, faclen) {
  sapply(labs, function(l) paste(strwrap(gsub(",", ", ", l), width = 30), collapse = "\n"))
}
# extra = 106: each box shows P(Endangered) and % of training rows
# nn = TRUE: node number above each box (matches the leaf column above)
rpart.plot(tree, type = 2, extra = 106, under = TRUE, fallen.leaves = TRUE,
           box.palette = "BuRd", roundint = FALSE, nn = TRUE,
           split.fun = wrap_labels, split.cex = 0.9, tweak = 1.15,
           main = paste("Endangered vs Threatened",
                        ifelse(USE_FYR_RECOMMENDATION, "- all features",
                               "- without fyr_recommendation")))