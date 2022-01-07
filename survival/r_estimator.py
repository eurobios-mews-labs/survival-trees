import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from sklearn.base import BaseEstimator, ClassifierMixin

from survival.tools import execution

tmp = "/tmp/" if "linux" in sys.platform else Path(os.environ["TEMP"])
tmp = os.fspath(tmp).replace("\\", "/")
path = os.path.abspath(__file__).replace(os.path.basename(__file__), "")


def install_if_needed(package_list: list):
    if len(package_list) == 1:
        package_list = f"('{package_list[0]}')"
    else:
        package_list = str(tuple(
            package_list
        ))
    r_cmd = """
            packages = c""" + package_list + """
            package.check <- lapply(
              packages,
              FUN = function(x) {
                if (!require(x, character.only = TRUE)) {
                  install.packages(x, dependencies = TRUE)
                  library(x, character.only = TRUE)
                }
              }
            )"""
    print(r_cmd)
    ro.r(r_cmd)


def install_ltrc_trees():
    if "linux" in sys.platform:
        r_cmd2 = """if (!require("LTRCtrees", character.only = TRUE)) {
                  install.packages("https://cran.r-project.org/src/contrib/Archive/LTRCtrees/LTRCtrees_1.1.0.tar.gz")
                }"""
    else:
        r_cmd2 = """if (!require("LTRCtrees", character.only = TRUE)) {
                  install.packages("https://cran.microsoft.com/snapshot/2017-08-01/bin/windows/contrib/3.4/LTRCtrees_0.5.0.zip")
                }"""

    ro.r(r_cmd2)


class REstimator(BaseEstimator):
    def __init__(self):
        self.__utils = rpackages.importr('utils')
        self.__utils.chooseCRANmirror(ind=1)
        self.__learn_name = "data.X"
        self.__test_name = "data.X"
        self._r_data_frame = pd.DataFrame()

    def _send_to_r_space(self, X, y=None):
        if y is not None:
            name = self.__learn_name
            data = pd.concat((X, y), axis=1)
        else:
            name = self.__test_name
            data = X
        with localconverter(ro.default_converter + pandas2ri.converter):
            self._r_data_frame = ro.conversion.py2rpy(data)
        ro.globalenv[name] = self._r_data_frame

    @staticmethod
    def _import_packages(list_package: List[str]):
        for package in list_package:
            rpackages.importr(package)

    @staticmethod
    def _get_from_r_space(list_object: List[str]):
        dict_result = {}
        for o in list_object:
            with localconverter(ro.default_converter + pandas2ri.converter):
                dict_result[o] = ro.conversion.rpy2py(ro.globalenv[o])
        return dict_result


class RandomForestSRC(REstimator, ClassifierMixin):
    def __init__(self, n_estimator=100):
        super().__init__()
        self.n_estimator = n_estimator
        self.name = "randomForestSRC"
        install_if_needed([self.name])
        self._import_packages(["randomForestSRC"])

    def fit(self, X, y: pd.DataFrame):
        """
        :param X: data frame
        :param y: 2D data set (time and status)
        :return:
        """
        if y.shape[1] != 2:
            raise ValueError("Target data should "
                             "be a dataframe with two columns")
        duration = y.columns[0]
        event = y.columns[1]
        self._send_to_r_space(X, y)
        ro.r(f"""
        forest.obj <- randomForestSRC::rfsrc(
            Surv({duration}, {event}) ~ .,
            data = data.X, 
            ntree = {self.n_estimator}, 
            tree.err=TRUE)""")
        ro.r("imp <- vimp(forest.obj)$importance")
        self.feature_importances_ = self._get_from_r_space(["imp"])["imp"]

    def predict(self, X) -> pd.DataFrame:
        self._send_to_r_space(X, y=None)
        ro.r("predict.obj <- predict(forest.obj, data.X)")
        ro.r("res <- predict.obj$survival")
        ro.r("times <- predict.obj$time.interest")
        getter = self._get_from_r_space(["res", "times"])
        prediction = getter["res"]
        times = getter["times"]
        return pd.DataFrame(prediction, columns=times, index=X.index)


class LTRCTrees(REstimator, ClassifierMixin):
    def __init__(
            self, max_depth=None,
            min_samples_leaf=None,
            get_dense_prediction=True,
            interpolate_prediction=True):
        super().__init__()
        install_if_needed(["survival", "LTRCtrees", "data.table",
                           "rpart", "Icens", "interval", 'stringi'])
        install_ltrc_trees()
        self._import_packages(["data.table", "LTRCtrees", "survival"])
        self.get_dense_prediction = get_dense_prediction
        self.interpolate_prediction = interpolate_prediction
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.__hash = "5B6if95PiV6z2j"

        str_ = "The target variable must"
        str_ += "be dataframe with following column order \n"

        print(str_ + "troncature, age_mort, mort")

    @execution.execution_time
    def _fit_old(self, X: pd.DataFrame, y: pd.DataFrame):
        y_copy = y.copy()
        y_copy.columns = ["troncature", "age_mort", "mort"]
        x = X.join(y_copy)
        x.to_csv(tmp + "/X", index=False)
        r_cmd = open(path + "/base_script.R").read()
        r_cmd = r_cmd.replace("{path}", "'" + tmp + "/X'")
        ro.r(r_cmd)

    @execution.execution_time
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        y_copy = y.copy()
        y_copy.columns = ["troncature", "age_mort", "mort"]
        self._send_to_r_space(X, y_copy)
        r_cmd = open(path + "/base_script_n.R").read()
        r_cmd = r_cmd % self.__param_r_setter()
        ro.r(r_cmd)
        self._id_run = ro.r("id.run")
        print(ro.r("result.ltrc.tree[[id.run]]"))

    def __param_r_setter(self):
        param = ""
        if self.min_samples_leaf is not None:
            param += "minbucket=%s, " % self.min_samples_leaf
        if self.max_depth is not None:
            param += "maxdepth=%s, " % self.max_depth
        if param == "":
            return ""
        else:
            return "control = rpart::rpart.control({param})".format(
                param=param[:-2])

    @execution.execution_time
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        self._send_to_r_space(X)
        ro.r(open(path + "/predict_n.R").read())
        km_mat = pd.DataFrame(
            columns=list(np.array(ro.r("time.stamp"))),
            index=X.index, dtype="float32")

        for k in list(ro.r("Keys")):
            subset = np.where(np.array(ro.r("data.X$ID")) == k)[0]

            curves = list(ro.r(
                "result$KMcurves[[{index}]]$surv".format(index=subset[0] + 1)))
            time = list(ro.r(
                "result$KMcurves[[{index}]]$time".format(index=subset[0] + 1)))
            km_mat.loc[km_mat.index[subset], np.array(time)] = curves
        if not self.get_dense_prediction:
            km_mat = km_mat.astype(pd.SparseDtype("float16", np.nan))
        if self.interpolate_prediction:
            km_mat = km_mat.T.fillna(method="pad").T
        return km_mat

    @execution.deprecated
    @execution.execution_time
    def _predict_old(self, X):
        self.__load()
        self.__test_name = "test"
        self.__path_to_test = tmp + "/test" + self.__hash
        X.to_csv(self.__path_to_test)
        r_cmd = open(path + "/predict.R").read()
        r_cmd = r_cmd.replace("{path}", "'" + self.__path_to_test + "'")
        ro.r(r_cmd)
        km_mat = pd.DataFrame(
            columns=list(np.array(ro.r("time.stamp"))),
            index=X.index, dtype="float32")
        for k in list(ro.r("Keys")):
            subset = np.where(np.array(ro.r("test$ID")) == k)[0]
            curves = list(ro.r(
                "result$KMcurves[[{index}]]$surv".format(index=subset[0] + 1)))
            time = list(ro.r(
                "result$KMcurves[[{index}]]$time".format(index=subset[0] + 1)))
            km_mat.loc[km_mat.index[subset], np.array(time)] = curves

        if not self.get_dense_prediction:
            km_mat = km_mat.astype(pd.SparseDtype("float16", np.nan))
        if self.interpolate_prediction:
            km_mat = km_mat.T.fillna(method="pad").T
        return km_mat

    @execution.deprecated
    def __save(self):
        ro.r('\
            saveRDS(rtree, file="{path}/{hash}rtree.Robject")\n\
            saveRDS(Keys.MM, file="{path}/{hash}keysMM.Robject")\n\
            saveRDS(List.KM, file="{path}/{hash}listKM.Robject")\n\
            saveRDS(List.Med, file="{path}/{hash}listMed.Robject")\n\
            '.format(path=tmp, hash=self.__hash))

    @execution.deprecated
    def __load(self):
        ro.r('\
            rtree <- readRDS(file="{path}/{hash}rtree.Robject")\n\
            Keys.MM <- readRDS(file="{path}/{hash}keysMM.Robject")\n\
            List.KM <- readRDS(file="{path}/{hash}listKM.Robject")\n\
            List.Med <- readRDS(file="{path}/{hash}listMed.Robject")\n\
            '.format(path=tmp, hash=self.__hash))


class RandomForestLTRC(ClassifierMixin):
    def __init__(self,
                 n_estimator=3, max_features=None,
                 max_depth=None, bootstrap=True, max_samples=0.8,
                 min_samples_leaf=None
                 ):
        self.estimators = {}
        self.select_feature = {}
        self.base = LTRCTrees
        self.bootstrap = bootstrap
        self.n_estimator = n_estimator
        self.max_samples = max_samples
        self.base_estimator = self.base(
            interpolate_prediction=False,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf
        )
        self.max_features = max_features

    @execution.execution_time
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        select_index = {}
        self.hashes = {}
        self.max_features = int(
            X.shape[1] / 3) if self.max_features is None else self.max_features
        for e in range(self.n_estimator):
            x_train, y_train = X.__deepcopy__(), y.__deepcopy__()
            if self.bootstrap:
                select_index[e] = np.random.choice(X.index, size=int(
                    self.max_samples * X.shape[0]))
                x_train, y_train = x_train.loc[select_index[e]], y.loc[
                    select_index[e]]

            self.select_feature[e] = np.random.choice(
                X.columns, size=int(self.max_features),
                replace=False)
            x_train = x_train.loc[:, self.select_feature[e]]

            self.hashes[e] = str(e)
            self.base_estimator.__hash = self.hashes[e]
            self.base_estimator.fit(x_train, y_train)

    def predict(self, X):
        result = {}
        for e in range(self.n_estimator):
            self.base_estimator.__hash = self.hashes[e]
            x_predict = X.loc[:, self.select_feature[e]]
            result[e] = self.base_estimator.predict(x_predict)
        return self.post_processing(result, X)

    @staticmethod
    def post_processing(result, X):
        data_dense = {}
        corresp = {}
        columns = pd.Series()
        all_times = np.unique(np.concatenate(
            [np.array(result[e].columns) for e in result.keys()]))
        for e in result.keys():
            corresp[e] = mat_corresp(result[e])
            index = corresp[e]["corresp"].unique()
            data_dense[e] = pd.DataFrame(columns=all_times, index=index)
            data_dense[e][result[e].columns] = result[e].loc[index].astype(
                float)
            data_dense[e][0] = 1
            data_dense[e] = data_dense[e][
                list(np.sort(data_dense[e].columns))].astype(float)
            data_dense[e] = data_dense[e].interpolate(
                method="linear", axis=1, limit_direction="forward")
            data_dense[e] = data_dense[e].drop(columns=0)

        res = pd.DataFrame(0, index=X.index, columns=all_times,
                           dtype="float32")
        res_n_sum = res.copy().astype(int)
        for t in all_times:
            columns.loc[t] = np.sum(
                [1 for e in result.keys() if t in data_dense[e].columns])
        columns.loc[0] = len(result.keys()) * 1000
        for e in result.keys():
            inds = np.unique(corresp[e].loc[:, "corresp"])
            for ind in inds:
                x_index = ind == corresp[e].loc[:, "corresp"]
                b = data_dense[e].index == ind
                res.loc[x_index, data_dense[e].columns] += data_dense[e].loc[
                    b].values
                res_n_sum.loc[x_index, data_dense[e].columns] += 1
        res = res / res_n_sum

        # res.iloc[:10].T.plot(legend=False, marker=".")
        return res


def mat_corresp(data):
    data = data.sort_values(list(np.sort(data.columns)))[
        list(np.sort(data.columns))]
    data_u = data.drop_duplicates()
    id = pd.DataFrame(index=data.index)
    id["corresp"] = np.nan
    id.loc[data_u.index, "corresp"] = data_u.index
    id = id.loc[data.index]
    id = id.fillna(method="pad")
    return id
