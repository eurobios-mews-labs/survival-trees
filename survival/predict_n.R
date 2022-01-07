# retrieve fit object (defined in base_script.R)


List.KM <- result.ltrc.tree[[id.run]]$km.curves
List.Med <- result.ltrc.tree[[id.run]]$median
Keys.MM <- result.ltrc.tree[[id.run]]$keys.mm

data.X$ID <- stats::predict(rtree, newdata = data.X, type = "vector")
Test.ID <- match(data.X$ID, Keys.MM[, 1])
Test.KM <- List.KM[Test.ID]
Test.Med <- unlist(List.Med[Test.ID])
result <- list(KMcurves = Test.KM, Medians = Test.Med)
Keys <- unique(data.X$ID)


time.stamp <- c()

for (k in Keys) {
  subset <- which(data.X$ID == k)
  time <- result$KMcurves[[subset[1]]]$time
  time.stamp <- c(time.stamp, time)
}
time.stamp <- sort(unique(time.stamp))


