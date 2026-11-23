# k-means clustering (Amanda)
#
# Groups listed species by how much protection work exists for them and when
# they were listed, then profiles each group. The outcomes (endangered status,
# five-year-review recommendation) and the descriptive categories (species
# group, region) are left out of the clustering on purpose, so we can ask
# afterwards how outcomes differ between clusters.

library(readxl)
library(cluster)

set.seed(160)
# read the data 
df <- as.data.frame(read_excel("new_data.xlsx", sheet = "model_data"))

# features for clusering
clust_vars <- c("listing_year", "is_plant", "is_population_segment",
                "has_recovery_doc", "rec_has_final_plan", "rec_n_documents",
                "rec_plan_revision", "has_five_year_review",
                "has_critical_habitat", "ch_total_acres", "ch_total_miles")
# create a scaled feature matrix for clustering
X <- df[, clust_vars]
# Log-transform acres and miles first, then scale all features to mean 0, SD 1.
X$ch_total_acres <- log1p(X$ch_total_acres)   # 0 to 120 million: log first
X$ch_total_miles <- log1p(X$ch_total_miles)
Xs <- scale(X)

# ---- Choose k: elbow (within-cluster SS) and average silhouette ---------------
ks  <- 2:10
# compute k-means for each k 
fits <- lapply(ks, function(k) kmeans(Xs, centers = k, nstart = 25, iter.max = 50))
# compute total within-cluster sum of squares
wss <- sapply(fits, function(f) f$tot.withinss)
# compute the distance between every pair of species
d   <- dist(Xs)
# compute average silhouette width for each k
sil <- sapply(fits, function(f) mean(silhouette(f$cluster, d)[, "sil_width"]))
# Print a table of the results
print(round(data.frame(k = ks, within_ss = wss, silhouette = sil), 3))

# Makes one image file 
png("figures/kmeans_choose_k.png", width = 1200, height = 550, res = 130)
# split the plotting area into 1 row, 2 columns
par(mfrow = c(1, 2))
# Left plot: elbow method (total within-cluster SS)
plot(ks, wss, type = "b", pch = 19, xlab = "k", ylab = "Total within-cluster SS",
     main = "Elbow")
# Right plot: average silhouette width
plot(ks, sil, type = "b", pch = 19, xlab = "k", ylab = "Average silhouette",
     main = "Silhouette")
# closes the PNG device and saves the file
dev.off()

# take the best k from the plot and assign cluster labels to the original data frame
K <- 4   # chosen from the plots above
km <- fits[[which(ks == K)]]
df$cluster <- km$cluster

# ---- Profile each cluster ------------------------------------------------------
# sorts the table of values and returns the top n values with their percentages
top <- function(x, n = 2) {
  t <- sort(table(x), decreasing = TRUE)[1:n]
  paste(sprintf("%s (%.0f%%)", names(t), 100 * t / length(x)), collapse = ", ")
}
# builds a profile of each cluster, including counts, percentages, and top categories 
profile <- do.call(rbind, lapply(sort(unique(df$cluster)), function(c) {
  s <- df[df$cluster == c, ]
  data.frame(
    cluster            = c,
    n                  = nrow(s),
    n_endangered       = sum(s$is_endangered == 1),
    n_threatened       = sum(s$is_endangered == 0),
    pct_endangered     = round(100 * mean(s$is_endangered), 1),
    pct_threatened     = round(100 * mean(s$is_endangered == 0), 1),
    median_listed      = median(s$listing_year),
    pct_plant          = round(100 * mean(s$is_plant), 1),
    pct_recovery_doc   = round(100 * mean(s$has_recovery_doc), 1),
    pct_final_plan     = round(100 * mean(s$rec_has_final_plan), 1),
    pct_5yr_review     = round(100 * mean(s$has_five_year_review), 1),
    pct_crit_habitat   = round(100 * mean(s$has_critical_habitat), 1),
    top_groups         = top(s$species_group),
    top_regions        = top(s$fws_lead_region)
  )
}))
options(width = 200)
print(profile)
# saves the profile to a CSV file for further analysis or reporting
write.csv(profile, "figures/cluster_profiles.csv", row.names = FALSE)

# Overall rates, for comparison with each cluster
cat(sprintf("\nAll species: %d | %d endangered (%.1f%%) | %d threatened (%.1f%%)\n",
            nrow(df), sum(df$is_endangered == 1), 100 * mean(df$is_endangered),
            sum(df$is_endangered == 0), 100 * mean(df$is_endangered == 0)))

# Bar chart: share threatened in each cluster vs the overall share
png("figures/kmeans_threatened_by_cluster.png", width = 900, height = 600, res = 130)
b <- barplot(profile$pct_threatened, names.arg = paste("Cluster", profile$cluster),
             ylab = "% Threatened", ylim = c(0, max(profile$pct_threatened) * 1.25),
             main = "Share of species listed as Threatened, by cluster")
# add a horizontal line for the overall share of threatened species
# can see which clusters have more or fewer threatened species than the overall average
abline(h = 100 * mean(df$is_endangered == 0), lty = 2, col = "red")
text(b, profile$pct_threatened, labels = paste0(profile$pct_threatened, "%"), pos = 3)
dev.off()

# Cluster centers in original units, to name the clusters
# give average values of each feature for each cluser
centers <- aggregate(X, by = list(cluster = df$cluster), FUN = mean)
print(round(centers, 2))

# A few example species from each cluster, nearest the center
# finds the 5 species closest to the cluster center for each cluster
for (c in sort(unique(df$cluster))) {
  i <- which(df$cluster == c)
  # compute the squared distance from each species in the cluster to the cluster center
  dist_c <- rowSums(sweep(Xs[i, , drop = FALSE], 2, km$centers[c, ])^2)
  # pick the 5 species with the smallest distance to the cluster center
  ex <- df$common_name[i[order(dist_c)[1:5]]]
  cat(sprintf("Cluster %d examples: %s\n", c, paste(ex, collapse = "; ")))
}

# 2-D picture of the clusters (first two principal components)
# principal component analysis (PCA) to reduce the dimensionality of the data for visualization
pc <- prcomp(Xs)
# plot the first two principal components, coloring points by their cluster assignment
png("figures/kmeans_clusters_pca.png", width = 1000, height = 800, res = 130)
# plot the first two principal components, coloring points by their cluster assignment
plot(pc$x[, 1:2], col = df$cluster, pch = 19, cex = 0.5,
     xlab = "PC1", ylab = "PC2", main = sprintf("k-means clusters (k = %d)", K))
legend("topright", legend = paste("Cluster", 1:K), col = 1:K, pch = 19, bty = "n")
dev.off()
