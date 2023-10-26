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
        self.df = df
    
    @abstractmethod
    def getSa(self) -> pd.Series:
        pass

class ScoreUpdateWinLoss(ABC):
    def getSa(self, df) -> pd.Series:
        return df['Win']

class ScoreUpdateRawPointsExp(ABC):
    def __init__(self, exp : float):
        self.exp = exp
    
    def getSa(self, df) -> pd.Series:
        sa = (df['PF'] ** self.exp) / ((df['PF'] ** self.exp) + (df['PA'] ** self.exp))
        return sa
        
class ScoreUpdateRawPointsLinear(ABC):
    def getSa(self, df) -> pd.Series:
        diff = df['PF'] - df['PA']
        sa = 0.0319 * diff + 0.5
        sa = np.where(sa < 0, 0, sa)
        sa = np.where(sa > 1, 1, sa)
        return sa
    
class EloRater():
    def __init__(self, base_df : pd.DataFrame, base_season : int = None):
        self.base_df = base_df
        self.base_season = base_season
        if not self.base_season:
            self.base_season = self.base_df['Season'].min()
        self._initYearOne()
    
    def _initYearOne(self):
        base_df = self.base_df.loc[self.base_df['Season'] == self.base_season, keep_cols].copy()
        EloYearRater(base_df)
        
class EloYearRater():
    def __init__(self, base_df : pd.DataFrame, year : int, currentRatings : pd.DataFrame = None, scoreUpdate : ScoreUpdate = None):
        self.year = year
        self.scoreUpdate = scoreUpdate
        if not self.scoreUpdate:
            self.scoreUpdate = ScoreUpdateWinLoss()
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
        print(self.base_df.head())
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
        last_date : pd.Timestamp = pd.Timestamp(self.year, 1, 1, 0)
        print(self.dateList)
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
            weekRatings = EloWeekRater(current_df, self.currentRatings, date, scoreUpdate = self.scoreUpdate)
            self.weekDict[date] = weekRatings
            self.currentRatings = weekRatings.getNewRatings()
            print(self.currentRatings.sort_values('Rating',ascending = False).head(5))
            print(self.currentRatings.shape)
            self.lossResults[date] = weekRatings.getNewLogLossResults()
            
class EloWeekRater():
    def __init__(self, base_df : pd.DataFrame, 
                 ratings : pd.DataFrame, 
                 date : pd.Timestamp, 
                 c : float = 400, 
                 k : float = 35,
                 scoreUpdate : ScoreUpdate = None):
        self.base_df = base_df
        self.initRatings = ratings
        self.date = date
        self.c = c
        self.k = k
        self.scoreUpdate = scoreUpdate
        if not self.scoreUpdate:
            self.scoreUpdate = ScoreUpdateWinLoss()
        self.df = self._generateNewRatings()
        
    def _generateNewRatings(self) -> pd.DataFrame:
        df = self.base_df.copy()
        df = df.merge(self.initRatings, on = 'Team', how = 'left', indicator = 'tm1')
        df = df.merge(self.initRatings, left_on = 'Opponent', right_on = 'Team', how = 'left', indicator = 'tm2')
        df['tm1'] = np.where(df['tm1'] == 'both', 1, 0)
        df['tm2'] = np.where(df['tm2'] == 'both', 1, 0)
        if len(df) > df['tm1'].sum() or len(df) > df['tm2'].sum():
            logger.error(f"Error processing elo for week {self.date}. Not all teams got rating, check names.")
            raise Exception("Error in EloWeekRater class - not all teams have ratings ")
        
        df['Rating_x_hfa'] = df['Rating_x'] + (0.5 - df['Location']) * 115
        df['qx'] = 10 ** (df['Rating_x_hfa'] / self.c)
        df['qy'] = 10 ** (df['Rating_y'] / self.c)
        df['ex'] = df['qx'] / (df['qx'] + df['qy'])
        df['sa'] = self.scoreUpdate.getSa(df)
        df['Rating_x_new'] = df['Rating_x'] + self.k * (self.scoreUpdate.getSa(df) - df['ex'])
        print(df[df['Team_x'] == 'Western Kentucky'])
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
    # tm_list = list(df['Team'].unique())
    # opp_list = list(df['Opponent'].unique())
    # big_list = list(set(tm_list + opp_list))
    # big_list.sort()
    er = EloYearRater(df, 2023, scoreUpdate = ScoreUpdateRawPointsExp(10))
    # Re-adjust so that each week you start from scratch and do X number of iterations over ratings
    # Iterative method seemingly wworks
    # But we need a way to rigorously implement and test it 
    er.beginRating()
    er.beginRating()
    er.beginRating()
    er.beginRating()
    er.beginRating()
    er.beginRating()
    er.beginRating()
    er.beginRating()
    er.beginRating()
    er.beginRating()
    er.beginRating()
    er.beginRating()
    
    last_date = er.dateList[-1]
    a = er.weekDict[last_date].newRatings.sort_values('Rating', ascending = False)
    a['Rank'] = a['Rating'].rank(ascending = False)
    print(a)
    a = pd.concat(list(er.lossResults.values()))
    print(log_loss(a['Win'],a['ex']))