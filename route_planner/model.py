import lightgbm as lgb
import numpy as np


class PlannerModel:
    @staticmethod
    def split(dfs, frac=0.8):
        if not dfs:
            raise ValueError("The dataframe tuple is empty")

        if any(len(df) != len(dfs[0]) for df in dfs):
            raise ValueError("The dataframes differ in size")

        msk = np.random.rand(len(dfs[0])) < frac

        return tuple([df.loc[msk] for df in dfs] + [df.loc[~msk] for df in dfs])

    @staticmethod
    def load_model(fn):
        return lgb.Booster(model_file=str(fn.resolve()))
