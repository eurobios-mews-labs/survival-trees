library(survival)
data <- survival::gbsg
data["start"] = data["age"]
data["stop"] = data["rfstime"] / 365.25 + data["age"]
data["event"] = data["status"]

data_test = data[data$pid > 1200,]
data_train = data[data$pid < 1200,]

Formula = Surv(start, stop, event) ~ age + meno + grade + size + pgr + er + hormon

## Fit an LTRCCIF on the time-invariant data, with mtry tuned with stepFactor = 3.
start_time <- Sys.time()
LTRCCIFobj = ltrcrrf(formula = Formula, data = data_train, ntree = 20, id=pid, mtry = NULL, nodesize=2)
end_time <- Sys.time()
end_time - start_time

tpnt = seq(min(data$start), max(data$stop), length.out = 1000)
# Set different upper time limits for each of the subjects
tau = seq(min(data$start), max(data$stop), length.out = length(unique(data_train$pid)))
## Obstain estimation at time points tpnt
Predobj = predictProb(object = LTRCCIFobj, time.eval = tpnt)
pbcobj = Surv(data_test$start, data_test$stop, data_test$event)
IBS = sbrier_ltrc(obj = pbcobj, id = data_test$pid, pred = Predobj, type = "IBS")
survival::concordance(LTRCCIFobj, data=data_train)
print(IBS)
