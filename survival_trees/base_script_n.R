
id.run <- stringi::stri_rand_strings(1, 20)

train.data <- data.X

Formula <- survival::Surv(troncature, age_mort, mort)~.
# rm(list = Filter(exists, c("rtree")))
rtree <- LTRCART(
    Formula,
    train.data,
    %s
    )
Formula[[3]] <-  1
ID <- stats::predict(rtree, type = "vector")
Keys <- unique(ID)
Keys.MM <- matrix(c(Keys, 1:length(Keys)), ncol = 2)
List.KM <- list()
List.Med <- list()

for (p in Keys) {
  subset <- train.data[ID == p,]
  KM <- survival::survfit(Formula, data = subset)
  Median <- utils::read.table(textConnection(
    utils::capture.output(KM)), skip = 2, header = TRUE)$median
  List.KM[[Keys.MM[Keys.MM[, 1] == p, 2] ]] <- KM
  List.Med[[Keys.MM[Keys.MM[, 1] == p, 2] ]] <- Median
}

if (!exists("result.ltrc.trees")){
    result.ltrc.tree <- hash()
}

result.ltrc.tree[[id.run]] <- hash(
    id=ID, keys=Keys, km.curves=List.KM, median=List.Med, keys.mm=Keys.MM,
    tree=rtree
)
var.importance <- rtree$variable.importance