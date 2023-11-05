import os
import pathlib
from typing import Dict, List, Any
from abc import ABC, abstractmethod
from joblib import load

import pandas as pd
import numpy as np
from .combine import process_years
from util.logger_config import setup_logger
import sklearn
from sklearn.metrics import log_loss

logger = setup_logger(__name__)
# Elo by winloss
# Elo by PF/PA
# Elo by YPP + Succcess Rate
# Home-field advantage


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

    
class EloRatingConfig():
    def __init__(self, **kwargs):
        self.c = kwargs.get('c', 400)
        self.k = kwargs.get('k', 35)
        self.scoreUpdate = kwargs.get('scoreUpdate', ScoreUpdateWinLoss())
        self.hfa = kwargs.get('hfa', 115)
        self.exp = None
        self.regWeight = kwargs.get('regWeight', None)
        if isinstance(self.scoreUpdate, ScoreUpdateRawPointsExp):
            self.exp = kwargs.get('exp', 10)
            self.scoreUpdate.exp = self.exp

    @property
    def dictForm(self) -> Dict[str, Any]:
        keys = ['c', 'k', 'scoreUpdate', 'hfa', 'exp', 'regWeight']
        vals = [self.c, self.k, self.scoreUpdate, self.hfa, self.exp, self.regWeight]
        print(keys)
        print(vals)
        return dict(zip(keys, vals))
    
class EloRater():
    def __init__(self, base_df : pd.DataFrame, eloRatingConfig : EloRatingConfig, min_season : int = None, max_season : int = None):
        self.base_df = base_df
        self.eloRatingConfig = eloRatingConfig

        self.min_season = min_season
        if not self.min_season:
            self.min_season = self.base_df['Season'].min()
        self.max_season = max_season
        if not self.max_season:
            self.max_season = self.base_df['Season'].max()
        
        self.lossResultsContainer : Dict[int, Dict[str, pd.DataFrame]] = {}
    
    def beginRating(self):
        for year in range(self.min_season, self.max_season + 1):
            er = EloYearRater(self.base_df, year, self.eloRatingConfig)
            er.beginRating()
            self.lossResultsContainer[year] = er.lossResults
    
    def getLossCalcColumns(self) -> pd.DataFrame:
        res = []
        for year, results in self.lossResultsContainer.items():
            a = pd.concat(list(results.values()))
            res.append(a)
        return pd.concat(res)
        
    def getResults(self, save : bool = False, savePath : str = None) -> float:
        lossCalcColumns = self.getLossCalcColumns()
        a = pd.to_numeric(lossCalcColumns['Win'], errors = 'coerce')
        b = pd.to_numeric(lossCalcColumns['ex'], errors = 'coerce')
        results = log_loss(a, b)
        if save:
            df = pd.DataFrame.from_dict({k : [v] for k, v in self.eloRatingConfig.dictForm.items()})
            df['min_season'] = self.min_season
            df['max_season'] = self.max_season
            df['samples'] = len(lossCalcColumns)
            df['score'] = results
            headerFlag = True
            if os.path.exists(savePath):
                headerFlag = False
            df.to_csv(savePath, mode = 'a', index = False, header = headerFlag)
        return results
    
class EloYearRater():
    def __init__(self, base_df : pd.DataFrame, year : int, eloRatingConfig : EloRatingConfig, currentRatings : pd.DataFrame = None):
        self.year = year        
        self.eloRatingConfig = eloRatingConfig
        self.base_df = self._limitFrame(base_df)
        self.dateList = self._getDateList()
        teamList = list(self.base_df['Team'].unique())
        self.currentRatings = self._initRatings(currentRatings, teamList)
        self.weekDict : Dict[str, EloWeekRater] = {}
        self.lossResults : Dict[str, pd.DataFrame] = {}
    
    def _initRatings(self, currentRatings : pd.DataFrame = None, teamList : List[str] = None) -> pd.DataFrame:
        if not currentRatings:
            currentRatings = pd.DataFrame.from_dict(dict(zip(teamList, [1500 for i in teamList])), orient = 'index').reset_index(drop=False)
            currentRatings.columns = ['Team','Rating']
        return currentRatings
    
    def _limitFrame(self, base_df) -> pd.DataFrame:
        keep_cols = self.eloRatingConfig.scoreUpdate.getKeepCols()
        return_df = base_df.loc[base_df['Season'] == self.year, keep_cols].copy()
        if len(return_df) == 0:
            logger.error("EloYearRater Error: Limiting dataframe yielded empty dataset. Re-check base dataset")
            raise Exception("EloYearRater Error: Limiting dataframe yielded empty dataset. Re-check base dataset")
        return return_df

    def _nextSaturday(self, date):
        while date.dayofweek != 6:
            date += pd.Timedelta(days = 1)
        return date

    def _getDateList(self) -> List[Any]:
        min_date = self.base_df['Date'].min()
        first_date = self._nextSaturday(min_date)
        max_date = self.base_df['Date'].max()
        start, end = sorted([first_date, max_date])
        
        # Create the list of dates incremented by one week
        date_list = []
        while start <= end:
            date_list.append(start)
            start += pd.Timedelta(weeks=1)
        date_list.append(start)
        return date_list

    def beginRatings(self, times : int):
        for i in range(times):
            self.beginRating()


    def beginRating(self):
        print(f"Beginning ratings for year {self.year}")
        last_date : pd.Timestamp = pd.Timestamp(self.year, 1, 1, 0)
        for date in self.dateList:
            current_df = self.base_df[(self.base_df['Date'] <= date) &
                         (self.base_df['Date'] > last_date)].copy()
            max_min_day_diff = (current_df['Date'].max() - current_df['Date'].min()).days
            
            if current_df['Date'].min() <= last_date or max_min_day_diff > 13:
                logger.error("EloYearRater Error: while creating weekly dataframe, minimum date is less than or equal to last week's value")
                raise Exception("EloYearRater Error: while creating weekly dataframe, minimum date is less than or equal to last week's value")
            if len(current_df[current_df['Team'].duplicated(keep=False)]) > 0:
                logger.error("EloYearRater Error: while creating weekly dataframe, found duplicates by team")
                print(current_df[current_df['Team'].duplicated(keep=False)])
                raise Exception("EloYearRater Error: while creating weekly dataframe, found duplicates by team")
            
            last_date = date
            weekRatings = EloWeekRater(current_df, self.currentRatings, date, config = self.eloRatingConfig)
            self.weekDict[date] = weekRatings
            self.currentRatings = weekRatings.getNewRatings()
            # print(self.currentRatings.sort_values('Rating',ascending = False).head(5))
            # print(self.currentRatings.shape)
            self.lossResults[date] = weekRatings.getNewLogLossResults()

class EloWeekRater():
    def __init__(self, base_df : pd.DataFrame, 
                 ratings : pd.DataFrame, 
                 date : pd.Timestamp, 
                 config : EloRatingConfig
                ):
        self.base_df = base_df
        self.initRatings = ratings
        self.date = date
        self.config = config
        self._loadConfig()
        self.df = self._generateNewRatings()
    
    def _loadConfig(self):
        self.c = self.config.c
        self.k = self.config.k
        self.scoreUpdate = self.config.scoreUpdate
        self.hfa = self.config.hfa
        self.exp = self.config.exp
        
    def _generateNewRatings(self) -> pd.DataFrame:
        df = self.base_df.copy()
        if len(df) == 0:
            return pd.DataFrame(columns = ['Team_x','Rating_x_new', 'Win','ex'])
        df = df.merge(self.initRatings, on = 'Team', how = 'left', indicator = 'tm1')
        df = df.merge(self.initRatings, left_on = 'Opponent', right_on = 'Team', how = 'left', indicator = 'tm2')
        df['tm1'] = np.where(df['tm1'] == 'both', 1, 0)
        df['tm2'] = np.where(df['tm2'] == 'both', 1, 0)
        if len(df) > df['tm1'].sum() or len(df) > df['tm2'].sum():
            logger.error(f"Error processing elo for week {self.date}. Not all teams got rating, check names.")
            raise Exception("Error in EloWeekRater class - not all teams have ratings ")
        
        df['Rating_x_hfa'] = df['Rating_x'] + (0.5 - df['Location']) * self.hfa
        df['Rating_y_hfa'] = df['Rating_y'] + (0.5 - (1 - df['Location'])) * self.hfa
        df['qx'] = 10 ** (df['Rating_x_hfa'] / self.c)
        df['qy'] = 10 ** (df['Rating_y'] / self.c)
        df['ex'] = df['qx'] / (df['qx'] + df['qy'])
        df['sa'] = self.scoreUpdate.getSa(df)
        df['Rating_x_new'] = df['Rating_x'] + self.k * (self.scoreUpdate.getSa(df) - df['ex'])
        return df
    
    def getNewRatings(self) -> pd.DataFrame:
        df = self.df
        merged = self.initRatings.merge(df[['Team_x','Rating_x_new']], left_on = 'Team', right_on = 'Team_x', how = 'left')
        merged ['Rating'] = np.where(merged['Rating_x_new'].notnull(), merged['Rating_x_new'], merged['Rating'])
        self.newRatings = merged[['Team', 'Rating']]
        return self.newRatings

    def getNewLogLossResults(self) -> pd.DataFrame:
        return self.df[['Win', 'ex']]

if __name__ == '__main__':
    path = pathlib.Path(__file__).parent.resolve()
    os.chdir(path)

    reg_path =  '../data/models/regression/PF_reg.joblib'
    
    # ScoreUpdateExpectedPoints('../data/models/regression/PF_reg.joblib')

    df = process_years(2016, 2024)
    
    c = 225
    k = 112.5
    # scoreUpdate = ScoreUpdateExpectedPoints(reg_path)
    scoreUpdate = ScoreUpdateExpectedPointsMerge(reg_path, 0.375)
    exp = None
    hfa = 60
    ratingConfig = {'c' : c, 'k' : k, 'scoreUpdate' : scoreUpdate, 'hfa' : hfa, 'exp' : exp, 'regWeight' : .375}
    eloRatingConfig = EloRatingConfig(**ratingConfig)
    # er = EloRater(df, eloRatingConfig, min_season = None, max_season = 2022)
    # er.beginRating()
    # print(er.getResults())
    
    eyr = EloYearRater(df, 2023, eloRatingConfig)
    eyr.beginRating()
    eyr.beginRating()
    eyr.beginRating()
    eyr.beginRating()
    eyr.beginRating()
    eyr.beginRating()
    eyr.beginRating()
    print(eyr.currentRatings.sort_values('Rating', ascending = False))

    
    # # 150 already done
    # for c in [200, 225, 250]:
    #     for k in [100, 125, 150]:
    #         for hfa in [30, 40, 50, 60, 70]:
    #             for scoreUpdate in [ScoreUpdateExpectedPointsMerge(reg_path, 0.5)]: #[ScoreUpdateRawPointsLinear(), ScoreUpdateRawPointsLinearNCAA()]:
    #                 exp_list = [None]
    #                 weight_list = [None]
    #                 if isinstance(scoreUpdate, ScoreUpdateRawPointsExp):
    #                     exp_list = [2, 5, 7.5, 10]
    #                 for exp in exp_list:
    #                     if exp is None and isinstance(scoreUpdate, ScoreUpdateRawPointsExp):
    #                         continue
    #                     ratingConfig = {'c' : c, 'k' : k, 'scoreUpdate' : scoreUpdate, 'hfa' : hfa, 'exp' : exp}
    #                     print(ratingConfig)
    #                     eloRatingConfig = EloRatingConfig(**ratingConfig)
    #                     er = EloRater(df, eloRatingConfig, min_season = None, max_season = 2022)
    #                     er.beginRating()
    #                     er.getResults(save = True, savePath = '../data/models/eloModelResults.csv')

    # for c in [200, 212.5, 225, 237.5, 250]:
    #     for k in [87.5, 100, 112.5, 125]:
    #         for hfa in [ 50, 60, 70]:
    #             for scoreUpdate in [ScoreUpdateExpectedPointsMerge(reg_path, 0.5)]: 
    #                 weight_list = [.125, .25, .375, .5, .625]
    #                 exp = None
    #                 for weight in weight_list:
    #                     scoreUpdate.regWeight = weight
    #                     ratingConfig = {'c' : c, 'k' : k, 'scoreUpdate' : scoreUpdate, 'hfa' : hfa, 'exp' : exp, 
    #                                     'regWeight' : weight}
    #                     print(ratingConfig)
    #                     eloRatingConfig = EloRatingConfig(**ratingConfig)
    #                     er = EloRater(df, eloRatingConfig, min_season = None, max_season = 2022)
    #                     er.beginRating()
    #                     er.getResults(save = True, savePath = '../data/models/eloModelResults.csv')