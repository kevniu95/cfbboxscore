import os
import pathlib
from typing import Dict, List, Any

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
    def __init__(self, base_df : pd.DataFrame, year : int, currentRatings : pd.DataFrame = None):
        self.year = year
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
        keep_cols = ['Date','Season', 'game_id','Team','Location','Opponent','Win']
        return base_df.loc[base_df['Season'] == self.year, keep_cols].copy()

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
            weekRatings = EloWeekRater(current_df, self.currentRatings, date)
            self.weekDict[date] = weekRatings
            self.currentRatings = weekRatings.getNewRatings()
            print(self.currentRatings.shape)
            self.lossResults[date] = weekRatings.getNewLossResults()
            
class EloWeekRater():
    def __init__(self, base_df : pd.DataFrame, ratings : pd.DataFrame, date : pd.Timestamp, c : float = 400, k : float = 32):
        self.base_df = base_df
        self.initRatings = ratings
        self.date = date
        self.c = c
        self.k = k
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
        df['Rating_x_new'] = df['Rating_x'] + self.k * (df['Win'] - df['ex'])
        return df
    
    def getNewRatings(self) -> pd.DataFrame:
        df = self.df
        merged = self.initRatings.merge(df[['Team_x','Rating_x_new']], left_on = 'Team', right_on = 'Team_x', how = 'left')
        merged ['Rating'] = np.where(merged['Rating_x_new'].notnull(), merged['Rating_x_new'], merged['Rating'])
        self.newRatings = merged[['Team', 'Rating']]
        return self.newRatings

    def getNewLossResults(self) -> pd.DataFrame:
        return self.df[['Win', 'ex']]



if __name__ == '__main__':
    path = pathlib.Path(__file__).parent.resolve()
    os.chdir(path)
    
    df = process_years(2016, 2023)
    # tm_list = list(df['Team'].unique())
    # opp_list = list(df['Opponent'].unique())
    # big_list = list(set(tm_list + opp_list))
    # big_list.sort()
    er = EloYearRater(df, 2016)
    er.beginRating()
    print(er.dateList)
    print(er.weekDict[pd.Timestamp('2017-01-15 00:00:00')].newRatings.sort_values('Rating', ascending = False))
    a = pd.concat(list(er.lossResults.values()))
    print(log_loss(a['Win'],a['ex']))