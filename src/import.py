import os
import pathlib
import time
import requests
import argparse
import json
from typing import Dict, Any, List
from bs4 import BeautifulSoup
import pandas as pd
from util.logger_config import setup_logger
from util.constants import GAMELOG_TABLE_COLUMNS
from util.config import Config
from util.util import send_get_request, get_max_year

'''
Have a logger for each individual module, have centralized logger config
Logger.exception() will propogate stacktrace, you can just use logger.exception() with an except:
   or other error-handling place
'''

logger = setup_logger(__name__)

class GameDataLoader():
    def __init__(self, config : Config = Config()):
        self.config = config
        self._processConfig()
    
    def _processConfig(self) -> None:
        try:
            sr_config = self.config.parse_section('sportsReference')
        except:
            logger.exception("Encountered error while parsing configuration in GameDataLoader initialization")
        self.base_site = sr_config['base_url']
        self.cfb_schools_suffix = sr_config['schools_suffix']
    
    def get_gamelog_links(self, path : str, year : int = 2023) -> Dict[str, str]:
        team_links = None
        if os.path.exists(path):
            with open(path, 'r') as read_content:
                team_links = json.load(read_content)
        else:
            logger.info(f"Couldn't find team links at file path: {path}. Re-scraping to get them...")
            resp = send_get_request(self.base_site + self.cfb_schools_suffix, None)
            team_links = self._get_team_links(resp)
            with open(path, 'w') as write_file:
                logger.info(f"Obtained team links and saving at file path: {path}")
                json.dump(team_links, write_file)
                
        gamelog_links = self._get_all_gamelog_links(team_links, year)
        if not gamelog_links:
            logger.error(f"Couldn't obtain game log links")
        return gamelog_links

    def _get_team_links(self, resp : requests.Response, max_year : int = None) -> Dict[str, str]:
        data = {}
        if not max_year:
            max_year = get_max_year() # Get this year's results by default

        if resp.ok:
            soup = BeautifulSoup(resp.text, 'html.parser')
            find_table_results = soup.find_all('table', id = 'schools')

            if find_table_results:
                rows = find_table_results[0].find_all('tr')
                for row in rows:
                    a = row.find('td', {'data-stat': 'year_max'})
                    if a and a.text == str(max_year):
                        b = row.find('a')
                        link = b['href']
                        data[b.text] = link
            else:
                logger.error("Function get_team_links() could not find table with id = 'schools'.")

        if len(data) == 0:
            logger.error("Function get_team_links() is returning an empty dictionary")
        return data

    def _get_all_gamelog_links(self, team_links : Dict[str, str], year : int = None) -> Dict[str, str]:
        newDict = {}
        if not year:
            year = get_max_year()

        for k, v in team_links.items():
            newDict[k] = f'{self.base_site}{v}{year}/gamelog/'
            
        if len(newDict) == 0:
            logger.error("Function get_all_gamelog_links() is returning an empty dictionary")
        return newDict

    def load_game_data(self, year : int, path : str, filename : str = 'gameResults.csv', modBy : int = 17) -> None:
        gamelog_links : Dict[str, str] = self.get_gamelog_links('../data/gamelogs/gamelog_links.json', year)
        filename_parts = filename.split('.')
        newFileName = f'{filename_parts[0]}_{year}.{filename_parts[1]}'
        self._load_game_data(gamelog_links, path = '../data/gamelogs', filename = newFileName, modBy = modBy)
    
    def _load_game_data(self, gamelog_links : Dict[str, str], path : str, filename : str = 'gameResults.csv', modBy : int = 17) -> None:
        ctr = 0
        all_dfs = []
        curr_list_of_dfs = []
        for k, v in gamelog_links.items():
            resp = send_get_request(v)
            soup = BeautifulSoup(resp.text, 'html.parser')
            off_table = soup.find_all('table', id = 'offense')
            if not off_table:
                logger.info(f"In load_game_data, couldn't find game log table for team {k}, skipping for now")
                continue

            rows = off_table[0].find_all('tr')
            data = []
            for row in rows:  # [1:] to skip the header
                if row.find('td', {'data-stat': 'date_game'}):
                    date_text = row.find('td', {'data-stat': 'date_game'})
                    if date_text.find('a'):
                        data.append(date_text.find('a')['href'])
            
            tab = pd.read_html(str(off_table), header = 1)[0]
            tab = tab[tab['Opponent'].notnull()].copy()
            tab['Team'] = k
            try:
                tab['game_id'] = pd.Series(data)
            except:
                logger.info(f"In load_game_data, game_id assignments don't match dataframe size, skipping this table for now")
                continue
            curr_list_of_dfs.append(tab)
            
            ctr += 1
            if ctr % modBy == 0:
                logger.info(f"Sleeping for 60 seconds then scraping gamelog data for new set of {modBy} teams...")
                all_dfs.extend(curr_list_of_dfs)
                curr_list_of_dfs.clear()
                time.sleep(61)

        all_dfs.extend(curr_list_of_dfs)        
        final_df = pd.concat(all_dfs)
        final_df.columns = GAMELOG_TABLE_COLUMNS
        final_df.to_csv(f'{path}/{filename}', index = False)


if __name__ == '__main__':
    path = pathlib.Path(__file__).parent.resolve()
    os.chdir(path)

    parser = argparse.ArgumentParser(description="Process a year argument.")
    parser.add_argument("year", type=int, help="Specify the year.")
    args = parser.parse_args()

    gameDataLoader = GameDataLoader()
    gameDataLoader.load_game_data(year = args.year, path = '../data/gamelogs/')