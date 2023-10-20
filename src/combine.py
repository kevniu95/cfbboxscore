import os
import pathlib
import pandas as pd
import numpy as np 
from typing import Callable

pd.set_option('display.max_rows', 500)
pd.set_option('display.expand_frame_repr', False)

def combine(year : int, path : str = '../data/gamelogs/', filename = 'gameResults'):
    """
    TODO: basic checks of combine function logic
    """
    final_path = os.path.join(path, filename) + f'_{year}.csv'
    df = pd.read_csv(final_path)
    df1 = df.merge(df, left_on = ['Date','game_id'], right_on = ['Date', 'game_id'], how = 'left')
    return df1

def process(df : pd.DataFrame):
    """
    TODO: Some checks at end of processing to make sure it's all good
    """
    sym_fields = ['Plays','Yards','TotalFirstDowns', 'PassTD','RushTD','Turnovers']
    x_fields = [i + '_x' for i in sym_fields]
    y_fields = [i + '_y' for i in sym_fields]
    
    df = df[~((df['PassFirstDowns_x'] == df['PassFirstDowns_y']) 
            & (df['RushFirstDowns_x'] == df['RushFirstDowns_y']) 
            & (df['PassAtt_x'] == df['PassAtt_y']) 
            & (df['RushAtt_x'] == df['RushAtt_y']) 
            & (df['PenaltyYards_x'] == df['PenaltyYards_y'])
            & (df['Fumbles_x'] == df['Fumbles_y'])
            & (df['Interceptions_x'] == df['Interceptions_y']))]
    
    df1 = df[['Date','game_id', 'Team_x','Location_x','Opponent_x','Result_x'] + x_fields + y_fields].copy()
    df1.rename(columns = {'Team_x' : 'Team',
                          'Opponent_x' : 'Opponent', 
                          'Location_x' : "Location", 
                          'Result_x' : "Result"}, inplace = True)
    df1[['PF','PA']] = df1['Result'].str.extract('.*\((.*)\).*')[0].str.split('-', expand = True)
    df1['TD_x'] = df1['PassTD_x'] + df1['RushTD_x']
    df1['TD_y'] = df1['PassTD_y'] + df1['RushTD_y']
    
    df1['YPP_x'] = df1['Yards_x'] / df1['Plays_x']
    df1['YPP_y'] = df1['Yards_y'] / df1['Plays_y']
    df1['SuccessRate_x'] = (df1['TotalFirstDowns_x'] + df1['TD_x']) / df1['Plays_x']
    df1['SuccessRate_y'] = (df1['TotalFirstDowns_y'] + df1['TD_y']) / df1['Plays_y']
    return_cols = ['Date', 'game_id', 'Team','Location','Opponent','PF','PA','YPP_x','YPP_y','SuccessRate_x','SuccessRate_y','Turnovers_x','Turnovers_y', 'Plays_x','Plays_y']
    
    return df1[return_cols]

def process_years(start_year : int, end_year : int, process_fx : Callable = process, combine_fx : Callable = combine) -> pd.DataFrame:
    # Like with Python's range, end_year won't be included
    game_results = []
    for year in range(start_year, end_year):
        df = process_fx(combine_fx(year))
        game_results.append(df)
    
    df = pd.concat(game_results)
    loc_dict = {'@' : 1, np.nan : 0, 'N' : 0.5}
    df['Location'] = df['Location'].replace(loc_dict)
    df['Win'] = np.where(df['PF'] > df['PA'], 1, 0)
    return df


# Elo by winloss
# Elo by PF/PA
# Elo by YPP + Succcess Rate
# Home-field advantage

if __name__ == '__main__':
    path = pathlib.Path(__file__).parent.resolve()
    os.chdir(path)

    df = process_years(2016, 2023)
    print(df.head())
    # sub_df = df[['Date','game_id','Team','Location','Opponent','PF','PA','Win']].copy()
    # sub_2016 = sub_df[pd.to_datetime(sub_df['Date']).dt.year == 2016].copy()
    # print(sub_2016.groupby('Date').count())
    
