# Populate the simple CSV for a given tournament
# Standard Library
import time
# 3rd Party
import pandas as pd
import numpy as np
import requests
# Repository Files
import _constants
import PastCollector
import WriteDataLocally

# Tournament we want all data from:
tourny_to_mine_from = 7


if __name__ == '__main__':
    df_to_write = pd.DataFrame()
    tournament_names = pd.read_csv(_constants.data_location + 'tournament_hash_table.csv')
    tournament_names = tournament_names[(tournament_names.leagueId == tourny_to_mine_from) &
                                        (tournament_names.published == 1)]

    # Sometimes we 504 - this is to catch those requests so we can try again later
    pending_requests = []
    failed = []

    # Get all tournament/match pairs (doing it this way to avoid code duplication in
    #   implementation of cached requests)
    for tournament in list(tournament_names['hash']):
        match_ids = PastCollector.get_hashes_in_tournament(tourny_to_mine_from, tournament)

        pending_requests += [(tournament, match_id, match_ids[match_id]) for match_id in match_ids]

    # Loop over matches (then games) and add rows to dataframe
    # While loop is to catch and reattempt 504'd attempts
    while pending_requests != []:
        # Count for progress indicator
        total_requests = len(pending_requests)

        for progress_index, (tournament, match, games_data) in enumerate(pending_requests):
            # Progress indicator
            if progress_index in [(i*total_requests)//10 for i in range(10)]:
                print('{0}/{1} matches attempted this cycle, {2} failed'.format(progress_index,
                                                                                total_requests,
                                                                                len(failed)))

            # Try/except to cache 504'd requests (which generally work later)
            # TODO find game hash more efficiently
            try:
                game_hashes = PastCollector.get_hashes_from_series(tournament, match)
            except ConnectionError:
                print('Tournament: {}, match: {} failed, will try again later'.format(tournament, match))
                failed += [(tournament, match, games_data)]
                continue

            # Actual stuff
            for game in games_data:
                game_hash = [hash_map['gameHash'] for hash_map in game_hashes
                             if hash_map['id'] == game[0]][0]
                new_df_rows = WriteDataLocally.simple_game_row(game[2], game[1], game_hash)
                new_df_rows['game_num'] = game[3]

                # Reorder columns a little
                cols = new_df_rows.columns.tolist()
                cols = cols[0:3] + cols[-1:] + cols[3:-1]
                new_df_rows = new_df_rows[cols]

                # Add to dataframe
                df_to_write = pd.concat([df_to_write, new_df_rows])

        if failed != []:
            time.sleep(30)  # Maybe it helps with the 504s
        pending_requests = failed[::-1]  # Reverse to attempt in order they occurred
        failed = []

    df_to_write.to_csv(_constants.data_location + 'simple_game_data_leagueId={}.csv'.format(tourny_to_mine_from))
