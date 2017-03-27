# File to populate a csv locally that has all the data about what tournaments
#   have been recently played
# N.B. when run initially, max(leagueId) = 38

# 3rd Party
import pandas as pd
import requests
# Repository Files
import _constants


# Number of requests to try where we see nothing before we give up
base_time_out_counter_value = 10  # Previously 100


if __name__ == '__main__':
    # Create dataframe that will become our csv and initialise variables to loop over
    output_tournament_data = pd.DataFrame(columns=['hash', 'leagueId', 'row_in_league', 'title'])
    leagueId = 0
    time_out_counter = base_time_out_counter_value

    while time_out_counter > 0:
        # When the counter hits zero, we stop searching
        attempt_req = requests.get(
            '{0}/scheduleItems?leagueId={1}'.format(_constants.api_base_esports1, leagueId))
        attempt_json = attempt_req.json()
        for row, tournament_data in enumerate(attempt_json['highlanderTournaments']):
            # Look through the json in the highlanderTournaments for entries in that list
            new_row = pd.Series({'hash': tournament_data['id'],
                                 'leagueId': leagueId,
                                 'row_in_league': row,
                                 'title': tournament_data['title'],
                                 'published': tournament_data['published']
                                 })
            # If we find something, add it to our output dataframe
            output_tournament_data = output_tournament_data.append(new_row, ignore_index=True)
            time_out_counter = base_time_out_counter_value
            print('Found in league {0}: {1}'.format(leagueId,
                                                    tournament_data['title']
                                                    ))
        leagueId += 1
        time_out_counter -= 1

    # Fix the int to float issue and write the dataframe to a csv
    output_tournament_data[['leagueId', 'row_in_league']] = output_tournament_data[['leagueId',
                                                                                    'row_in_league']].astype(int)
    output_tournament_data.to_csv(_constants.data_location + 'tournament_hash_table.csv')
    print('Data written')
