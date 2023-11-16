# Use Google Custom Search API to search for top 10 results of query "Michigan Ohio State football game 2021" where all results are before 2021-11-27
from googleapiclient.discovery import build
import pathlib
import os
import datetime
import pandas as pd
from util.config import Config
from .combine import process_years
from typing import List

SEARCH_API_KEY = Config().parse_section('google')['api_key']
SEARCH_CSE_ID = Config().parse_section('google')['cse_id']
    

def getSearchUrls(query : str, start_date : str, end_date : str) -> List[str]:
    print(query)
    
    # Set up the Custom Search API client
    service = build("customsearch", "v1", developerKey=SEARCH_API_KEY)

    # Execute the search
    res = service.cse().list(q=query, cx=SEARCH_CSE_ID, sort=f"date:r:{start_date}:{end_date}").execute()
    # res = service.cse().list(q=query, cx=SEARCH_CSE_ID, sort=f"date").execute()
    # Print the top 10 search results
    for i, result in enumerate(res['items'][:10]):
        print()
        print(result)
        print("Result {}: {}".format(i+1, result['link']))
    return [result['link'] for result in res['items'][:10]]

if __name__ == '__main__':
    path = pathlib.Path(__file__).parent.resolve()
    os.chdir(path)

    df = process_years(2016, 2024)
    df_sub = df[(df['Team'] == 'Michigan') & df['Date'].dt.year.isin([2017])]
    for i, row in df_sub.iterrows():
        start_date = (row['Date'] - pd.Timedelta(days=6)).strftime('%Y%m%d')
        end_date = (row['Date'] - pd.Timedelta(days=1)).strftime('%Y%m%d')
        opponent = row['Opponent']
        queryText = f"{row['Team']} {row['Opponent']} football preview"
        print(start_date)
        print(end_date)
        resList = getSearchUrls(queryText, start_date, end_date)
        print(resList)
        if i > 1:
            break
        # print(f"Michigan vs {opponent} on {row['Date'].strftime('%Y-%m-%d')}")
        # print(f"Searching for articles between {start_date} and {end_date}")    
    