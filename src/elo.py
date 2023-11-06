import os
import pathlib
from typing import Dict, List, Any
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from .combine import process_years
from util.logger_config import setup_logger
from sklearn.metrics import log_loss
from src.scoreUpdate import *


logger = setup_logger(__name__)
# Elo by winloss
# Elo by PF/PA
# Elo by YPP + Succcess Rate
# Home-field advantage

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
    
    def get_keep_cols(self):
        return self.scoreUpdate.getKeepCols()
    
class EloRater():
    def __init__(self, base_df : pd.DataFrame, eloRatingConfig : EloRatingConfig, min_season : int = None, max_season : int = None):
        self.base_df = base_df
        self.eloRatingConfig = eloRatingConfig
        
        self.min_season = min_season or self.base_df['Season'].min()
        self.max_season = max_season or self.base_df['Season'].max()
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
        
    def getScoredResults(self, save : bool = False, savePath : str = None) -> float:
        lossCalcColumns = self.getLossCalcColumns()
        lossCalcColumns[['Win', 'ex']] = lossCalcColumns[['Win', 'ex']].apply(pd.to_numeric, errors='coerce')
        results = log_loss(lossCalcColumns['Win'], lossCalcColumns['ex'])
        if save:
            df = pd.DataFrame.from_dict({k : [v] for k, v in self.eloRatingConfig.dictForm.items()})
            df['min_season'] = self.min_season
            df['max_season'] = self.max_season
            df['samples'] = len(lossCalcColumns)
            df['score'] = results
            headerFlag = not os.path.exists(savePath)
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
    
    def _initRatings(self, currentRatings: pd.DataFrame = None, teamList: List[str] = None) -> pd.DataFrame:
        return currentRatings if currentRatings is not None else pd.DataFrame({'Team': teamList, 'Rating': 1500})
    
    def _limitFrame(self, base_df) -> pd.DataFrame:
        keep_cols = self.eloRatingConfig.get_keep_cols()
        return_df = base_df.loc[base_df['Season'] == self.year, keep_cols].copy()
        if len(return_df) == 0:
            logger.error("EloYearRater Error: Limiting dataframe yielded empty dataset. Re-check base dataset")
            raise Exception("EloYearRater Error: Limiting dataframe yielded empty dataset. Re-check base dataset")
        return return_df

    def _getDateList(self) -> List[Any]:
        min_date = self.base_df['Date'].min()
        first_date = self._nextSaturday(min_date)
        max_date = self.base_df['Date'].max()
        start, end = sorted([first_date, max_date])
        
        # Create the list of dates incremented by one week
        date_list = pd.date_range(start=start, end=end, freq='W-SAT').tolist()
        return date_list

    @staticmethod
    def _nextSaturday(date):
        return (date + pd.Timedelta((5 - date.dayofweek) % 7 + 1, unit='d')).normalize()

    def beginRatings(self, times : int):
        for i in range(times):
            self.beginRating()

    def beginRating(self):
        print(f"Beginning ratings for year {self.year}")
        prv_date : pd.Timestamp = pd.Timestamp(self.year, 1, 1, 0)
        for date in self.dateList:
            current_df = self.base_df[self.base_df['Date'].between(prv_date, date, inclusive = 'right')].copy()  
            max_min_day_diff = (current_df['Date'].max() - current_df['Date'].min()).days
            
            if current_df['Date'].min() <= prv_date or max_min_day_diff > 13:
                logger.error("EloYearRater Error: while creating weekly dataframe, minimum date is less than or equal to last week's value")
                raise Exception("EloYearRater Error: while creating weekly dataframe, minimum date is less than or equal to last week's value")
            if len(current_df[current_df['Team'].duplicated(keep=False)]) > 0:
                logger.error("EloYearRater Error: while creating weekly dataframe, found duplicates by team")
                raise Exception("EloYearRater Error: while creating weekly dataframe, found duplicates by team")
            
            prv_date = date
            weekRatings = EloWeekRater(current_df, self.currentRatings, date, config = self.eloRatingConfig)
            self.weekDict[date] = weekRatings
            self.currentRatings = weekRatings.getNewRatings()
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
        self.newRatingDf = self._generateNewRatings()
    
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
        
        # Reassign self.base_df to be new values for updated columns of df, but limited to columns that were originally in self.base_df
        self.base_df.loc[df.index, self.base_df.columns.intersection(df.columns)] = df.loc[:, self.base_df.columns.intersection(df.columns)]
        return df
    
    @property
    def newRatings(self) -> pd.DataFrame:
        if hasattr(self, '_newRatings_cache'):
            return self._newRatings_cache
        else:
            return self._newRatings()

    def _newRatings(self) -> pd.DataFrame:
        # Private implementation of getNewRatings() that returns a dataframe
        merged = self.initRatings.merge(self.newRatingDf[['Team_x', 'Rating_x_new']], left_on='Team', right_on='Team_x', how='left')
        merged['Rating'] = merged['Rating_x_new'].fillna(merged['Rating'])
        self._newRatings_cache = merged[['Team', 'Rating']]
        return self._newRatings_cache
    
    def getNewLogLossResults(self) -> pd.DataFrame:
        return self.newRatingDf[['Win', 'ex']]

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