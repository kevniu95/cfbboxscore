import os
from configparser import ConfigParser
from typing import Dict

class Config():
    def __init__(self, filename : str = '../../config.ini'):
        self.filename : str = self._getConfigPath(filename)
        # create a parser
        self.parser : ConfigParser = self._setupParser()
    
    def _getThisPath(self) -> str:
        return os.path.dirname(os.path.abspath(__file__))

    def _getConfigPath(self, filename : str):
        thisPath = self._getThisPath()
        return os.path.join(thisPath, filename)
    
    def _setupParser(self) -> ConfigParser:
        parser = ConfigParser()
        # read config file
        parser.read(self.filename)
        return parser
    
    def parse_section(self, section : str) -> Dict[str, str]:
        out = {}
        if self.parser.has_section(section):
            params = self.parser.items(section)
            for param in params:
                out[param[0]] = param[1]
        else:
            raise Exception('Section {0} not found in the {1} file'.format(section, self.filename))
        return out