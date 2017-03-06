# Standard Library
import json
# 3rd Party
import pandas as pd
import numpy as np
import requests


# Constants - TODO move to config file to generalise
esports_api_base = 'http://2015.na.lolesports.com/api/'
# Also found 'http://api.lolesports.com/api/v2/' while looking at ws TODO investigate usefulness
# Also maybe v1 instead of v2


def get_week_hashes(tournament, week):
    """Get hashes for all games that week in given tournament"""
    pass


def get_data_from_hash(hash):
    """Given a hash for a match, get all information about match"""
    pass

