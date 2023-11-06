from abc import ABC, List, abstractmethod
import pandas as pd
from util.logger_config import setup_logger
import numpy as np
from joblib import load

logger = setup_logger(__name__)


class ScoreUpdate(ABC):
    def __init__(self):
        self.keep_cols = ['Date','Season', 'game_id','Team','Location','Opponent','Win', 'PF', 'PA']
    
    def getKeepCols(self) -> List[str]:
        return self.keep_cols

    @abstractmethod
    def _getSa(self, df) -> pd.Series:
        pass

    def getSa(self, df) -> pd.Series:
        self.df = df
        if not self.isReady():
            logger.error(f"{self.__class__.__name__} class is not ready!")
            raise Exception("Error in ScoreUpdate function, class is not ready!")
        return self._getSa(df)

    def isReady(self) -> bool:
        if isinstance(self.df, pd.DataFrame):
            return True
        return False

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

class ScoreUpdateWinLoss(ScoreUpdate):
    def _getSa(self, df) -> pd.Series:
        return df['Win']
    
    def isReady(self) -> bool:
        if super().isReady() and 'Win' in self.df.columns:
            return True
        return False
    
class ScoreUpdateRawPointsExp(ScoreUpdate):
    def __init__(self, exp : float = None):
        self.exp = exp

    def _getSa(self, df) -> pd.Series:
        sa = (df['PF'] ** self.exp) / ((df['PF'] ** self.exp) + (df['PA'] ** self.exp))
        return sa

    def isReady(self) -> bool:
        if super().isReady() and 'PF' in self.df.columns and 'PA' in self.df.columns and self.exp is not None:
            return True
        return False
            
class ScoreUpdateRawPointsLinear(ScoreUpdate):
    def _getSa(self, df) -> pd.Series:
        diff = df['PF'] - df['PA']
        sa = 0.0319 * diff + 0.5
        sa = np.where(sa < 0, 0, sa)
        sa = np.where(sa > 1, 1, sa)
        return sa
    
    def isReady(self) -> bool:
        if super().isReady() and 'PF' in self.df.columns and 'PA' in self.df.columns:
            return True
        return False

class ScoreUpdateRawPointsLinearNCAA(ScoreUpdate):
    def _getSa(self, df) -> pd.Series:
        diff = df['PF'] - df['PA']
        sa = 0.0255 * diff + 0.5
        sa = np.where(sa < 0, 0, sa)
        sa = np.where(sa > 1, 1, sa)
        return sa

    def isReady(self) -> bool:
        if super().isReady() and 'PF' in self.df.columns and 'PA' in self.df.columns:
            return True
        return False

class ScoreUpdateExpectedPoints(ScoreUpdate):
    def __init__(self, regModelPath : str):
        super().__init__()
        self.keep_cols.extend(['YPP_x','YPP_y','SuccessRate_x','SuccessRate_y','Plays_x','Plays_y'])
        self.regModel : sklearn.pipeline.Pipeline = self._loadRegression(regModelPath)
    
    def _loadRegression(self, regModelPath : str):
        try:
            return load(regModelPath)
        except:
            logger.error("ScoreUpdateExpectedPoints Error: Error loading regression model into ScoreUpdate by expected points regression")
            raise Exception("ScoreUpdateExpectedPoints Error: Error loading regression model into ScoreUpdate by expected points regression")
            
    def _getExpectedDiff(self, df):
        df_copy = df.copy()
        reg_cols = ['YPP_x','SuccessRate_x', 'YPP_y','SuccessRate_y', 'Plays_x', 'Plays_y']
        df_copy['PF_exp'] = self.regModel.predict(df[reg_cols])
        
        df_copy[['YPP_x', 'YPP_y']] = df_copy[['YPP_y', 'YPP_x']]
        df_copy[['SuccessRate_x', 'SuccessRate_y']] = df_copy[['SuccessRate_y', 'SuccessRate_x']]
        df_copy[['Plays_x', 'Plays_y']] = df_copy[['Plays_y', 'Plays_x']]
        df_copy['PA_exp'] = self.regModel.predict(df_copy[reg_cols])
        diff = df_copy['PF_exp'] - df_copy['PA_exp']
        return diff
    
    def _getSa(self, df) -> pd.Series:
        # logger.info("Generating new sa for expected points ScoreUpdate")
        diff = self._getExpectedDiff(df)
        sa = 0.0255 * diff + 0.5
        sa = np.where(sa < 0, 0, sa)
        sa = np.where(sa > 1, 1, sa)
        return sa
    
    def isReady(self) -> bool:
        reg_cols = ['YPP_x','SuccessRate_x', 'YPP_y','SuccessRate_y', 'Plays_x', 'Plays_y']
        if super().isReady() and set(reg_cols).issubset(self.df.columns):
            return True
        return False

class ScoreUpdateExpectedPointsMerge(ScoreUpdateExpectedPoints):
    def __init__(self, regModelPath : str, regWeight : float):
        self.regWeight = regWeight
        super().__init__(regModelPath)
    
    def _getSa(self, df) -> pd.Series:
        temp_sa = super()._getSa(df)
        og_diff = df['PF'] - df['PA']
        diff = (temp_sa * (self.regWeight)) + (og_diff * (1 - self.regWeight))
        sa = 0.0319 * diff + 0.5
        sa = np.where(sa < 0, 0, sa)
        sa = np.where(sa > 1, 1, sa)
        return sa
