import os
import pathlib
import time
import requests
import json
from requests import Session, Request
from typing import Dict, Any, List
from bs4 import BeautifulSoup
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
from util.logger_config import setup_logger
from util.constants import GAMELOG_TABLE_COLUMNS
from util.config import Config

logger = setup_logger(__name__)

def send_get_request(url : str, params : Dict[str, Any] = None) -> requests.Response:
    try:
        s = Session()
        req = Request('GET', url , params = params).prepare()
        r = s.send(req)
    except Exception as e:
        logger.exception(f"Failed to create or send request: {e}")
    if not r.ok:
        logger.error(f"Response returned from {url} was not OK")
    return r

def get_max_year() -> int:
    six_months_ago = date.today() - relativedelta(months = 8)
    return six_months_ago.year

def get_team_links(resp : requests.Response, max_year : int = None) -> Dict[str, str]:
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

def get_all_gamelog_links(team_links : Dict[str, str], year : int = None) -> Dict[str, str]:
    newDict = {}
    if not year:
        year = get_max_year()

    for k, v in team_links.items():
        newDict[k] = f'{base_site}{v}{year}/gamelog/'
        
    if len(newDict) == 0:
        logger.error("Function get_all_gamelog_links() is returning an empty dictionary")
    return newDict

def load_game_data(gamelog_links : Dict[str, str], path : str, filename : str = 'gameResults.csv', modBy : int = 17) -> None:
    ctr = 0
    all_dfs = []
    curr_list_of_dfs = []
    for k, v in gamelog_links.items():
        resp = send_get_request(v)
        soup = BeautifulSoup(resp.text, 'html.parser')
        off_table = soup.find_all('table', id = 'offense')
        if not off_table:
            logger.error(f"In load_game_data, couldn't find game log table for team {k}, skipping for now")
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
            logger.error(f"In load_game_data, game_id assignments don't match dataframe size, skipping this table for now")
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

def get_gamelog_links(path : str) -> Dict[str, str]:
    if os.path.exists(path):
        with open(path, 'r') as read_content:
            logger.info(f"Found and retrieving gamelog links from file path: {path}")
            return json.load(read_content)
    else:
        logger.info(f"Couldn't find gamelog links at file path: {path}. Re-scraping to get them...")
        resp = send_get_request(base_site + cfb_schools_suffix, None)
        team_links = get_team_links(resp)
        gamelog_links = get_all_gamelog_links(team_links)
        if gamelog_links:
            logger.info(f"Obtained game links and saving at file path: {path}")
            with open(path, 'w') as write_file:
                json.dump(team_links, write_file)
        else:
            logger.error(f"Couldn't scrape game log links")
            return {}

if __name__ == '__main__':
    path = pathlib.Path(__file__).parent.resolve()
    os.chdir(path)

    sr_config = Config().parse_section('sportsReference')
    base_site = sr_config['base_url']
    cfb_schools_suffix = sr_config['schools_suffix']

    get_gamelog_links('../data/gamelog_links.json')
    # save_path = '../data/'
    # load_game_data(gamelog_links, save_path, modBy = 15)