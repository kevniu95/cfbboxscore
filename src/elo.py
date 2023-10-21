import os
import pathlib
from typing import Dict, List, Any

import pandas as pd
import numpy as np
from .combine import process_years
from util.logger_config import setup_logger

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
        self.currentRatings = currentRatings
        if not self.currentRatings:
            self.currentRatings = pd.DataFrame.from_dict(dict(zip(teamList, [1500 for i in teamList])), orient = 'index').reset_index(drop=False)
            self.currentRatings.columns = ['Team','Rating']
        weekDict : Dict[str, EloWeekRater] = dict(zip(teamList, [None for i in teamList]))
    
    def _limitFrame(self, base_df) -> pd.DataFrame:
        keep_cols = ['Date','Season', 'game_id','Team','Location','Opponent','Win']
        return base_df.loc[base_df['Season'] == self.year, keep_cols].copy()

    def _getDateList(self) -> List[Any]:
        max_date = self.base_df['Date'].max()
        grouped_dates = self.base_df.groupby('Date').count().reset_index()
        first_date = grouped_dates[grouped_dates['game_id'] > 50].reset_index(drop = True).loc[0,'Date']
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
        for date in self.dateList[:1]:
            current_df = self.base_df[(self.base_df['Date'] <= date) &
                         (self.base_df['Date'] >= last_date)].copy()
            last_date = date
            EloWeekRater(current_df, self.currentRatings, date)

class EloWeekRater():
    def __init__(self, base_df : pd.DataFrame, ratings : pd.DataFrame, date : pd.Timestamp, c : float = 400, k : float = 32):
        self.base_df = base_df
        self.ratings = ratings
        self.date = date
        self.c = c
        self.k = k
        self._createInitTable()
        
    def _createInitTable(self) -> None:
        df = self.base_df.copy()
        df = df.merge(self.ratings, on = 'Team', how = 'left', indicator = 'tm1')
        df = df.merge(self.ratings, left_on = 'Opponent', right_on = 'Team', how = 'left', indicator = 'tm2')
        df['tm1'] = np.where(df['tm1'] == 'both', 1, 0)
        df['tm2'] = np.where(df['tm2'] == 'both', 1, 0)
        if len(df) > df['tm1'].sum() or len(df) > df['tm2'].sum():
            logger.error(f"Error processing elo for week {self.date}. Not all teams got rating, check names.")
            raise Exception("Error in EloWeekRater class - not all teams have ratings ")
        
        df['qx'] = 10 ** (df['Rating_x'] / self.c)
        df['qy'] = 10 ** (df['Rating_y'] / self.c)
        df['ex'] = df['qx'] / (df['qx'] + df['qy'])
        df['Rating_x_new'] = df['Rating_x'] + self.k * (df['Win'] - df['ex'])

        print(df.head())
        # print(self.ratings)
        



        

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