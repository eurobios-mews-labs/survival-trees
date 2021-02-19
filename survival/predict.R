# retrieve fit object (defined in base_script.R)

test <- read.csv({path}, sep = ",")

test$ID <- stats::predict(rtree, newdata = test, type = "vector")
Test.ID <- match(test$ID, Keys.MM[, 1])
Test.KM <- List.KM[Test.ID]
Test.Med <- unlist(List.Med[Test.ID])
result <- list(KMcurves = Test.KM, Medians = Test.Med)
Keys <- unique(test$ID)


time.stamp <- c()

for (k in Keys) {
  subset <- which(test$ID == k)
  time <- result$KMcurves[[subset[1]]]$time
  time.stamp <- c(time.stamp, time)
}
time.stamp <- sort(unique(time.stamp))
#
#M <- matrix(, ncol = length(time.stamp), nrow = (length(result$KMcurves)))
#Y <- data.frame(M)
#Y[,1] <- 1
#colnames(Y) <- time.stamp
#
#for (k in Keys) {
#  subset <- which(test$ID == k)
#  result.temp <- result$KMcurves[[subset[1]]]
#  km.curve <- result.temp$surv
#  time.list <- result.temp$time
#  Y[subset, unlist(strsplit(toString(time.list), ", "))] <- km.curve
#}
#
#Y <- round(Y, digits = 4)

