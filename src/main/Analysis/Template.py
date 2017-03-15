# 3rd Party imports
import pandas as pd
import numpy as np
# Repository Files
import _constants

if __name__ == '__main__':
    df = pd.read_csv(_constants.data_location + 'simple_game_data_leagueId=6.csv')
    print(df)
