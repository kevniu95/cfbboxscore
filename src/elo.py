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
        self.its = (kwargs.get('its', 1))

    @property
    def dictForm(self) -> Dict[str, Any]:
        keys = ['c', 'k', 'scoreUpdate', 'hfa', 'exp', 'regWeight', 'its']
        vals = [self.c, self.k, self.scoreUpdate, self.hfa, self.exp, self.regWeight, self.its]
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
            eyr = EloYearRater(self.base_df, year, self.eloRatingConfig)
            eyr.beginRating()
            self.lossResultsContainer[year] = eyr.lossResults
    
    def beginRatings(self, iterations : int = 1):
        for year in range(self.min_season, self.max_season + 1):
            print(f"Processing year {year}")
            eyr = EloYearRater(self.base_df, year, self.eloRatingConfig)
            eyr.beginRatings(iterations)
            self.lossResultsContainer[year] = eyr.lossResults
    
    def getLossCalcColumns(self) -> pd.DataFrame:
        res = []
        for year, results in self.lossResultsContainer.items():
            a = pd.concat(list(results.values()))
            res.append(a)
        result_df = pd.concat(res)
        result_df[['Win', 'ex']] = result_df[['Win', 'ex']].apply(pd.to_numeric, errors='coerce')
        return result_df
        
    def getScoredResults(self, save : bool = False, savePath : str = None) -> float:
        lossCalcColumns = self.getLossCalcColumns()
        results = log_loss(lossCalcColumns['Win'], lossCalcColumns['ex'])
        lossCalcColumns1 = lossCalcColumns[lossCalcColumns['weekNumber'] <= 10].copy()
        results1 = log_loss(lossCalcColumns1['Win'], lossCalcColumns1['ex'])
        lossCalcColumns2 = lossCalcColumns[lossCalcColumns['weekNumber'] > 10].copy()
        results2 = log_loss(lossCalcColumns2['Win'], lossCalcColumns2['ex'])
        if save:
            df = pd.DataFrame.from_dict({k : [v] for k, v in self.eloRatingConfig.dictForm.items()})
            df['min_season'] = self.min_season
            df['max_season'] = self.max_season
            df['samples'] = len(lossCalcColumns)
            df['firstHalfScore'] = results1
            df['secondHalfScore'] = results2
            df['score'] = results
            headerFlag = not os.path.exists(savePath)
            df.to_csv(savePath, mode = 'a', index = False, header = headerFlag)
        return results, results1, results2
    
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
        first_date = self._nextSunday(min_date)
        max_date = self.base_df['Date'].max() + pd.Timedelta(days = 6)  
        start, end = sorted([first_date, max_date])
        
        # Create the list of dates incremented by one week
        date_list = pd.date_range(start=start, end=end, freq='W-SUN', inclusive = 'both').tolist()
        return date_list

    @staticmethod
    def _nextSunday(date):
        while date.dayofweek != 6:
            date += pd.Timedelta(days = 1)
        return date

    def beginRatings(self, times : int):
        for dt in self.dateList:
            for i in range(times):
                # Limit self.dateList to all dates from beginning up to and including dt
                dateList = self.dateList[:self.dateList.index(dt) + 1]
                # Somehow limit dateList to date in question
                self.beginRating(dateList, i)

    def beginRating(self, dateList : List[Any], iterationNum : int = 0):
        """
        What is the scope of this function?

        Iterate over each date in dateList, starting from first
         and upon each iteration:
        1. Grab subset of games related to that week as indicated by date
        2. Send game subset and current ratings to EloWeekRater
        3. EloWeekRater does following
            a. Updates currentRatings and 
            b. Produces some df of predictions vs results

        So in theory, should be able to:
        - Go through steps 1-3 arbitrary number of times- call it x - for any one week,
          then move on to next week

        And it would be consistent rating system

        Question to think through - how would we save out the LogLossResults
        """
        prv_date : pd.Timestamp = pd.Timestamp(self.year, 1, 1, 0)
        weekNum = len(dateList) - 1
        if iterationNum == 0:
            prv_date = dateList[-2] if len(dateList) > 1 else prv_date
            dateList = dateList[-1:]
        # print('prv_date and dateList values are now the following...')
        # print(prv_date)
        # print(dateList)
        for date in dateList:
            current_df = self.base_df[self.base_df['Date'].between(prv_date, date, inclusive = 'right')].copy()  
            max_min_day_diff = (current_df['Date'].max() - current_df['Date'].min()).days
            
            if current_df['Date'].min() <= prv_date or max_min_day_diff > 13:
                logger.error("EloYearRater Error: while creating weekly dataframe, minimum date is less than or equal to last week's value")
                raise Exception("EloYearRater Error: while creating weekly dataframe, minimum date is less than or equal to last week's value")
            if len(current_df[current_df['Team'].duplicated(keep=False)]) > 0:
                logger.error("EloYearRater Error: while creating weekly dataframe, found duplicates by team")
                print(current_df[current_df['Team'].duplicated(keep=False)])
                raise Exception("EloYearRater Error: while creating weekly dataframe, found duplicates by team")
            
            prv_date = date
            weekRatings = EloWeekRater(current_df, self.currentRatings, date, weekNum, config = self.eloRatingConfig)
            self.weekDict[date] = weekRatings
            self.currentRatings = weekRatings.currentRatings
            if iterationNum == 0:
                # print(f"Saving results for date {date}")
                self.lossResults[date] = weekRatings.getNewLogLossResults()
                # print(self.lossResults[date].head())
    
    def getScores(self, team: str):
        scoreUpdate = self.eloRatingConfig.scoreUpdate 
        df = self.base_df
        df = df[df['Team'] == team].copy()
        df[['PF_exp', 'PA_exp']] = scoreUpdate.getExpectedScore(df)
        print(df[['Date', 'Location','Opponent', 'PF', 'PA', 'PF_exp', 'PA_exp']])
        
class EloWeekRater():
    def __init__(self, base_df : pd.DataFrame, 
                 ratings : pd.DataFrame, 
                 date : pd.Timestamp, 
                 dateIndex : int,
                 config : EloRatingConfig
                ):
        self.base_df = base_df # Shouldn't need to be updated
        self.final_df = None # Updated at end of each getNewRatings() call
        self.currentRatings = ratings # Updated with getNewRatings()call
        self.date = date
        self.dateIndex = dateIndex
        self.config = config
        self._loadConfig()
        self.generateNewRatings()
        
    def _loadConfig(self):
        self.c = self.config.c
        self.k = self.config.k
        self.scoreUpdate = self.config.scoreUpdate
        self.hfa = self.config.hfa
        self.exp = self.config.exp
        self.innerItsTodo = self.config.its
        self.innerItsDone = 0

    def generateNewRatings(self) -> None:
        for i in range(self.innerItsTodo):
            ''' On each iteration, leave base_df alone, update final_df'''
            self._generateNewRatings()  
        
    def _generateNewRatings(self) -> pd.DataFrame:
        """
        Create a generateNewRatings function
        1. Merge current ratings with base_df
        2. Calculate new rating
        3. Return new ratings and new base_df
            4. New ratings and base_df must be in same format from start to finish
        """
        df = self.base_df.copy()
        if len(df) == 0:
            self.final_df = pd.DataFrame(columns = ['Team_x','Rating_x_new', 'Win','ex', 'weekNumber', 'weekDate'])
            return
        df = df.merge(self.currentRatings, left_on = 'Team', right_on = 'Team', how = 'left', indicator = 'tm1')
        df = df.merge(self.currentRatings, left_on = 'Opponent', right_on = 'Team', how = 'left', indicator = 'tm2')
        
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
        self.currentRatings = self._updateCurrentRatings(df)

        if self.innerItsDone == 0:
            self.final_df = df[['Team_x','Rating_x_new', 'Win','ex']].copy()
        if self.innerItsDone + 1 == self.innerItsTodo:
            self.final_df['Rating_x_new'] = df['Rating_x_new']
            self.final_df['weekNumber'] = self.dateIndex
            self.final_df['weekDate'] = self.date
        self.innerItsDone += 1
        return df
    
    def _updateCurrentRatings(self, df : pd.DataFrame) -> pd.DataFrame:
        merged = self.currentRatings.merge(df[['Team_x', 'Rating_x_new']], left_on = 'Team', right_on = 'Team_x', how='left')
        merged['Rating'] = merged['Rating_x_new'].fillna(merged['Rating'])
        return merged[['Team', 'Rating']]    

    def getNewLogLossResults(self) -> pd.DataFrame:
        if 'Win' not in self.final_df.columns or 'ex' not in self.final_df.columns:
            logger.error(f"Error getting log loss results for week {self.date}. 'Win' or 'ex' not in base_df")
            raise Exception("Error in EloWeekRater class - Win or ex not in base_df")
        return self.final_df[['weekDate', 'weekNumber','Win', 'ex']]

if __name__ == '__main__':
    path = pathlib.Path(__file__).parent.resolve()
    os.chdir(path)

    reg_path =  '../data/models/regression/PF_reg.joblib'
    df = process_years(2016, 2024)
    c = 225
    k = 40
    regWeight = 0.5
    scoreUpdate = ScoreUpdateExpectedPointsMerge(reg_path, regWeight)
    exp = None
    hfa = 55
    ratingConfig = {'c' : c, 'k' : k, 'scoreUpdate' : scoreUpdate, 'hfa' : hfa, 'exp' : exp, 'regWeight' : regWeight, 'its' : 2}
    eloRatingConfig = EloRatingConfig(**ratingConfig)
    # er = EloRater(df, eloRatingConfig, min_season = None, max_season = 2022)
    # er.beginRating()
    # print(er.getResults())
    # scoreUpdateList = [ScoreUpdateExpectedPoints(reg_path), 
    #                ScoreUpdateRawPointsLinear(), 
    #                ScoreUpdateExpectedPointsMerge(reg_path, 0.375)]

    eyr = EloYearRater(df, 2023, eloRatingConfig)
    # eyr.beginRatings(2)
    # print(eyr.currentRatings.sort_values('Rating', ascending = False))
    eyr.getScores('Alabama')


    

    # eloRatingConfig = EloRatingConfig(**ratingConfig)
    # er = EloRater(df, eloRatingConfig, min_season = 2023, max_season = 2023)
    # er.beginRatings(2)
    # print(er.getLossCalcColumns())
    # print(er.getScoredResults(save= False))
    
    # # 150 already done
    # for c in [150, 200, 250]:
    #     for k in [100, 200, 300]:
    #         for hfa in [25, 50, 75]:
    #             for scoreUpdate in scoreUpdateList: #[ScoreUpdateRawPointsLinear(), ScoreUpdateRawPointsLinearNCAA()]:
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

    # 187.5, 200 done already
    # for c in [200, 225, 250, 275]:
    #     print(c)
    #     for k in [40, 45, 50]: # WHen finished explore more below 75
    #         print(k)
    #         for hfa in [55, 67.5, 75]: # WHen finished, converge on 50
    #             print(hfa)
    #             for its in [1, 2]:  
    #                 print(its)
    #                 for scoreUpdate in scoreUpdateList: 
    #                     print(scoreUpdate)
    #                     if isinstance(scoreUpdate, ScoreUpdateExpectedPointsMerge):
    #                         weight_list = [.5] # Come back to weight adjustments after toggling rest of parameters
    #                     else:
    #                         weight_list = [None]
    #                     for weight in weight_list:
    #                         scoreUpdate.regWeight = weight
    #                         ratingConfig = {'c' : c, 'k' : k, 'scoreUpdate' : scoreUpdate, 'hfa' : hfa, 'exp' : exp, 
    #                                         'regWeight' : weight, 'its' : its}
    #                         print(ratingConfig)
    #                         eloRatingConfig = EloRatingConfig(**ratingConfig)
    #                         er = EloRater(df, eloRatingConfig, min_season = None, max_season = 2022)
    #                         er.beginRatings(its)
    #                         er.getScoredResults(save = True, savePath = '../data/models/eloModelResults_its.csv')