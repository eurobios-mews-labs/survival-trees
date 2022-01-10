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

r_session = ro.r


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
    r_session(r_cmd)


def install_ltrc_trees():
    if "linux" in sys.platform:
        r_cmd2 = """if (!require("LTRCtrees", character.only = TRUE)) {
                  install.packages("https://cran.r-project.org/src/contrib/Archive/LTRCtrees/LTRCtrees_1.1.0.tar.gz")
                }"""
    else:
        r_cmd2 = """if (!require("LTRCtrees", character.only = TRUE)) {
                  install.packages("https://cran.microsoft.com/snapshot/2017-08-01/bin/windows/contrib/3.4/LTRCtrees_0.5.0.zip")
                }"""

    r_session(r_cmd2)


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
        r_session(f"""
        forest.obj <- randomForestSRC::rfsrc(
            Surv({duration}, {event}) ~ .,
            data = data.X, 
            ntree = {self.n_estimator}, 
            tree.err=TRUE)""")
        r_session("imp <- vimp(forest.obj)$importance")
        self.feature_importances_ = self._get_from_r_space(["imp"])["imp"]

    def predict(self, X) -> pd.DataFrame:
        self._send_to_r_space(X, y=None)
        r_session("predict.obj <- predict(forest.obj, data.X)")
        r_session("res <- predict.obj$survival")
        r_session("times <- predict.obj$time.interest")
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
                           "rpart", "Icens", "interval", 'stringi', "hash"])
        install_ltrc_trees()
        self._import_packages(["data.table", "LTRCtrees", "survival", 'hash'])
        self.get_dense_prediction = get_dense_prediction
        self.interpolate_prediction = interpolate_prediction
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.__hash = "id.run"

        str_ = "The target variable must"
        str_ += "be dataframe with following column order \n"

        print(str_ + "troncature, age_mort, mort")

    def _fit_old(self, X: pd.DataFrame, y: pd.DataFrame):
        y_copy = y.copy()
        y_copy.columns = ["troncature", "age_mort", "mort"]
        x = X.join(y_copy)
        x.to_csv(tmp + "/X", index=False)
        r_cmd = open(path + "/base_script.R").read()
        r_cmd = r_cmd.replace("{path}", "'" + tmp + "/X'")
        r_session(r_cmd)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        y_copy = y.copy()
        y_copy.columns = ["troncature", "age_mort", "mort"]
        self._send_to_r_space(X, y_copy)
        r_cmd = open(path + "/base_script_n.R").read()
        r_cmd = r_cmd % self.__param_r_setter()
        r_session(r_cmd)
        self._id_run = str(self._get_from_r_space(["id.run"])["id.run"][0])
        self.results_ = r_session("result.ltrc.tree")
        ro.r("gc()")

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

    def __get_prediction_data(self, X: pd.DataFrame):
        self._send_to_r_space(X)
        ro.globalenv["result.ltrc.tree"] = self.results_
        r_cmd_str = open(path + "/predict_n.R").read().replace(
            "id.run", "'%s'" % self._id_run
        )
        r_session(r_cmd_str)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        self.__get_prediction_data(X)
        km_mat = pd.DataFrame(
            columns=list(np.array(r_session("time.stamp"))),
            index=X.index, dtype="float16")

        for k in list(r_session("Keys")):
            subset = np.where(np.array(r_session("data.X$ID")) == k)[0]

            curves = list(r_session(
                "result$KMcurves[[{index}]]$surv".format(index=subset[0] + 1)))
            time = list(r_session(
                "result$KMcurves[[{index}]]$time".format(index=subset[0] + 1)))
            km_mat.loc[km_mat.index[subset], np.array(time)] = curves
        if not self.get_dense_prediction:
            km_mat = km_mat.astype(pd.SparseDtype("float16", np.nan))
        if self.interpolate_prediction:
            km_mat = km_mat.T.fillna(method="pad").T
        ro.r("gc()")
        return km_mat

    def predict_curves(self, X: pd.DataFrame) -> tuple:
        self.__get_prediction_data(X)
        all_times = list(np.array(r_session("time.stamp")))
        curves = pd.DataFrame(columns=all_times, index=range(len(
            list(r_session("Keys")))), dtype="float32")
        indexes = pd.Series(index=X.index, dtype="int64")
        for i, k in enumerate(list(r_session("Keys"))):
            subset = np.where(np.array(r_session("data.X$ID")) == k)[0]
            curve = np.array(r_session(
                "result$KMcurves[[{index}]]$surv".format(index=subset[0] + 1)))
            time = np.array(r_session(
                "result$KMcurves[[{index}]]$time".format(index=subset[0] + 1)))
            curve = pd.Series(curve, index=time, dtype="float32").reindex(
                all_times)
            curves.loc[i] = curve.fillna(method="pad")
            indexes.iloc[subset] = i
        return curves, indexes

    @execution.deprecated
    @execution.execution_time
    def _predict_old(self, X):
        self.__load()
        self.__test_name = "test"
        self.__path_to_test = tmp + "/test" + self.__hash
        X.to_csv(self.__path_to_test)
        r_cmd = open(path + "/predict.R").read()
        r_cmd = r_cmd.replace("{path}", "'" + self.__path_to_test + "'")
        r_session(r_cmd)
        km_mat = pd.DataFrame(
            columns=list(np.array(r_session("time.stamp"))),
            index=X.index, dtype="float32")
        for k in list(r_session("Keys")):
            subset = np.where(np.array(r_session("test$ID")) == k)[0]
            curves = list(r_session(
                "result$KMcurves[[{index}]]$surv".format(index=subset[0] + 1)))
            time = list(r_session(
                "result$KMcurves[[{index}]]$time".format(index=subset[0] + 1)))
            km_mat.loc[km_mat.index[subset], np.array(time)] = curves

        if not self.get_dense_prediction:
            km_mat = km_mat.astype(pd.SparseDtype("float16", np.nan))
        if self.interpolate_prediction:
            km_mat = km_mat.T.fillna(method="pad").T
        return km_mat

    @execution.deprecated
    def __save(self):
        r_session('\
            saveRDS(rtree, file="{path}/{hash}rtree.Robject")\n\
            saveRDS(Keys.MM, file="{path}/{hash}keysMM.Robject")\n\
            saveRDS(List.KM, file="{path}/{hash}listKM.Robject")\n\
            saveRDS(List.Med, file="{path}/{hash}listMed.Robject")\n\
            '.format(path=tmp, hash=self.__hash))

    @execution.deprecated
    def __load(self):
        r_session('\
            rtree <- readRDS(file="{path}/{hash}rtree.Robject")\n\
            Keys.MM <- readRDS(file="{path}/{hash}keysMM.Robject")\n\
            List.KM <- readRDS(file="{path}/{hash}listKM.Robject")\n\
            List.Med <- readRDS(file="{path}/{hash}listMed.Robject")\n\
            '.format(path=tmp, hash=self.__hash))


class RandomForestLTRC(ClassifierMixin):
    def __init__(self,
                 n_estimator=3, max_features=None,
                 max_depth=None, bootstrap=True, max_samples=1,
                 min_samples_leaf=None, base_estimator=LTRCTrees,
                 ):
        self.__select_feature = {}
        self.base_estimator = base_estimator
        self.bootstrap = bootstrap
        self.n_estimator = n_estimator
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.base_estimator = self.base_estimator(
            interpolate_prediction=False,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf
        )
        self.max_features = max_features

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):

        self.__hashes = {}
        self.results_ = {}
        self.max_features = int(
            X.shape[1] / 3) if self.max_features is None else self.max_features
        self.max_features = max(2, self.max_features)
        self.max_features = min(X.shape[1], self.max_features)
        for e in range(self.n_estimator):
            x_train, y_train = self.__bootstrap(X, y)
            self.__select_feature[e] = np.random.choice(
                X.columns, size=int(self.max_features),
                replace=False)
            x_train = x_train.loc[:, self.__select_feature[e]]
            self.base_estimator.fit(x_train, y_train)
            self.results_[e] = self.base_estimator.results_
            self.__hashes[e] = self.base_estimator._get_from_r_space([
                "id.run"])["id.run"][0]

    def __bootstrap(self, X: pd.DataFrame, y: pd.DataFrame):
        if self.bootstrap:
            select_index = np.random.choice(X.index, size=int(
                self.max_samples * X.shape[0]))
            x_train, y_train = X.loc[select_index], y.loc[
                select_index]
            return x_train, y_train
        return X, y

    def predict(self, X, return_type="dense"):
        self.fast_predict(X)
        if return_type == "dense":
            return pd.merge(self.nodes_, self.km_estimates_,
                            left_on="curve_index",
                            right_index=True).set_index("x_index").drop(
                columns=["curve_index"]).loc[X.index]

    def fast_predict(self, X: pd.DataFrame) -> pd.DataFrame:
        result = {}
        for e in range(self.n_estimator):
            x_predict = X.loc[:, self.__select_feature[e]]
            self.base_estimator.results_ = self.results_[e]
            self.base_estimator._id_run = self.__hashes[e]
            result[e] = self.base_estimator.predict_curves(x_predict)
        self.km_estimates_, self.nodes_ = self.post_processing_fast(result, X)

    def _predict_old(self, X):
        result = {}
        for e in range(self.n_estimator):
            x_predict = X.loc[:, self.__select_feature[e]]
            self.base_estimator.results_ = self.results_[e]
            self.base_estimator._id_run = self.__hashes[e]
            result[e] = self.base_estimator.predict(x_predict)
        return self.post_processing(result, X)

    @staticmethod
    def post_processing_fast(result, X):
        nodes = pd.DataFrame(False, index=X.index, columns=result.keys())
        all_times = [result[e][0].columns for e in result.keys()]
        all_times = np.unique(np.sort(np.concatenate(all_times)))

        # calculate node/run association
        for e in result.keys():
            data = result[e][1]
            nodes.loc[data.index, e] = data.values

        unique_nodes = nodes.drop_duplicates()
        unique_curves = pd.DataFrame(
            0, columns=all_times, index=unique_nodes.index,
            dtype="float32")
        unique_curves = unique_curves.loc[
                        :, ~unique_curves.columns.duplicated()]
        unique_curves_mask = pd.DataFrame(
            0, columns=unique_curves.columns,
            index=unique_nodes.index, dtype="Int8")

        # calculate km curves for every combination possible
        for c in unique_nodes.columns:
            data = result[c][0].loc[unique_nodes[c]]
            data = data.T.reindex(
                unique_curves.columns).fillna(
                method="ffill").fillna(
                method="bfill").T
            unique_curves += data.values
            unique_curves_mask[data.columns] += (~data.isna()
                                                 ).astype(int).values

        unique_curves /= unique_curves_mask
        nodes = nodes.reset_index()
        unique_nodes = unique_nodes.reset_index()
        on = list(result.keys())
        nodes = pd.merge(nodes, unique_nodes,
                         on=on).drop(columns=on)
        nodes.columns = ["x_index", "curve_index"]
        return unique_curves, nodes

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
