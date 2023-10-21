from typing import Dict, Any, List
import requests
from requests import Session, Request
from datetime import date, datetime
from dateutil.relativedelta import relativedelta

from util.logger_config import setup_logger


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