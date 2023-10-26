import os
import pathlib
from typing import Dict, List, Any
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from .combine import process_years
from util.logger_config import setup_logger
from sklearn.metrics import log_loss

logger = setup_logger(__name__)
# Elo by winloss
# Elo by PF/PA
# Elo by YPP + Succcess Rate
# Home-field advantage


class ScoreUpdate(ABC):
    def __init__(self):
        pass
    
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
        if isinstance(df, pd.DataFrame):
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
            
class ScoreUpdateRawPointsLinear(ABC):
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
    
class EloRatingConfig():
    def __init__(self, **kwargs):
        self.dictForm = kwargs
        self.c = kwargs.get('c', 400)
        self.k = kwargs.get('k', 35)
        self.scoreUpdate = kwargs.get('scoreUpdate', ScoreUpdateWinLoss())
        self.hfa = kwargs.get('hfa', 115)
        self.exp = None
        if isinstance(self.scoreUpdate, ScoreUpdateRawPointsExp):
            self.exp = kwargs.get('exp', 10)
            self.scoreUpdate.exp = self.exp
    
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
        results = log_loss(lossCalcColumns['Win'], lossCalcColumns['ex'])
        if save:
            df = pd.DataFrame.from_dict({k : [v] for k, v in eloRatingConfig.dictForm.items()})
            df['min_season'] = self.min_season
            df['max_season'] = self.max_season
            df['samples'] = len(lossCalcColumns)
            df['score'] = results
            df.to_csv(savePath, mode = 'a', index = False)
        return results
    
class EloYearRater():
    def __init__(self, base_df : pd.DataFrame, year : int, eloRatingConfig : EloRatingConfig, currentRatings : pd.DataFrame = None):
        self.year = year        
        self.base_df = self._limitFrame(base_df)
        self.eloRatingConfig = eloRatingConfig
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
        keep_cols = ['Date','Season', 'game_id','Team','Location','Opponent','Win', 'PF', 'PA']
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

    def beginRating(self):
        print(f"Beginning ratings for year {self.year}")
        last_date : pd.Timestamp = pd.Timestamp(self.year, 1, 1, 0)
        for date in self.dateList:
            print(date)
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
        df = df.merge(self.initRatings, on = 'Team', how = 'left', indicator = 'tm1')
        df = df.merge(self.initRatings, left_on = 'Opponent', right_on = 'Team', how = 'left', indicator = 'tm2')
        df['tm1'] = np.where(df['tm1'] == 'both', 1, 0)
        df['tm2'] = np.where(df['tm2'] == 'both', 1, 0)
        if len(df) > df['tm1'].sum() or len(df) > df['tm2'].sum():
            logger.error(f"Error processing elo for week {self.date}. Not all teams got rating, check names.")
            raise Exception("Error in EloWeekRater class - not all teams have ratings ")
        
        df['Rating_x_hfa'] = df['Rating_x'] + (0.5 - df['Location']) * self.hfa
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
    
    df = process_years(2016, 2024)
    ratingConfig = {'c' : 400, 'k' : 35, 'scoreUpdate' : ScoreUpdateWinLoss(), 'hfa' : 115, 'exp' : 10}
    eloRatingConfig = EloRatingConfig(**ratingConfig)
    er = EloRater(df, eloRatingConfig, min_season = None, max_season = 2022)
    er.beginRating()
    print(er.getResults(save = True, savePath = '../data/models/eloModelResults.csv'))