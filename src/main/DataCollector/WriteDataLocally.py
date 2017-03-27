# Standard Library
import re
# import json
# 3rd Party
import pandas as pd
import numpy as np
import requests
# Repository Files
import _constants
import PastCollector

# TODO refactor into one class when all made


# Unused
def map_position_to_name(num):
    """Returns the position name from an integer 0 to 4"""
    try:
        return ['TOP', 'JUNGLE', 'MID', 'ADC', 'SUPPORT'][num]
    except IndexError:
        raise ValueError('Please input a number 0, 1, 2, 3 or 4')


def access_json_by_tuple(json_object, keys):
    """Accesses a nested dictionary (json) with the keys in a tuple"""
    for k in keys:
        json_object = json_object[k]
    return json_object


def simple_game_row(game_realm, game_id, game_hash):
    """Extracts features from the acs game data"""
    game_data = PastCollector.get_data_from_hash(game_realm, game_id, game_hash)
    # Collect data  that I see as important for both teams and put them into a dict to
    #   later turn into two pd.Series objects that we can use to update a pd.DataFrame
    data_by_team = [{}, {}]

    # TODO fix for amateur and ensure that this causes no issues for pro
    team_names = [re.match('(.*?) ', game_data['participantIdentities'][5*i]['player']['summonerName']).group(1)
                  for i in range(2)]

    # Keys we want that we don't need to apply any processing to
    # Keys are tuples that give their path in the json object
    unfiltered_keys = {}
    for side in (0, 1):
        unfiltered_keys[side] = {('participants', i, 'stats'): [
                               # Basic stats
                               'kills', 'assists', 'deaths', 'goldEarned',
                               'totalMinionsKilled', 'champLevel', 'turretKills',
                               # Items
                               'item1', 'item2', 'item3', 'item4', 'item5', 'item6',
                               # Damage dealt/taken
                               'physicalDamageDealt', 'physicalDamageDealtToChampions',
                               'trueDamageDealt', 'trueDamageDealtToChampions',
                               'magicDamageDealt', 'magicDamageDealtToChampions',
                               'totalDamageDealt', 'totalDamageDealtToChampions',
                               'magicalDamageTaken', 'physicalDamageTaken',
                               'trueDamageTaken', 'totalHeal',
                               # Sprees
                               'doubleKills', 'tripleKills', 'quadraKills',
                               'pentaKills', 'largestKillingSpree',
                               # Wards
                               'wardsPlaced', 'visionWardsBoughtInGame',
                               'wardsKilled',
                               # Misc
                               'firstBloodAssist', 'firstTowerAssist',
                               'firstInhibitorAssist',
                               'inhibitorKills',
                               'longestTimeSpentLiving',
                               'totalTimeCrowdControlDealt',
                               'neutralMinionsKilledEnemyJungle',
                               'neutralMinionsKilledTeamJungle',
                           ] for i in range(5*side, 5*side + 5)
                           }

    # Adding more keys but by looping as they all look similar
        for i in range(5*side, 5*side + 5):
            unfiltered_keys[('participants', i)] = ['spell1Id', 'spell2Id']
            for deep_key_val in ['creepsPerMinDeltas', 'csDiffPerMinDeltas', 'goldPerMinDeltas',
                                 'damageTakenPerMinDeltas', 'xpPerMinDeltas',
                                 'damageTakenDiffPerMinDeltas', 'xpDiffPerMinDeltas']:
                unfiltered_keys[('participants', i, 'timeline', deep_key_val)] = [
                    '0-10', '10-20', '20-30', '30-end'
                ]

    # Flattening the json with the keys we want
    for team in range(2):
        # Some custom details (keys that we need to apply some processing to)
        if game_data['teams'][team]['win'] == 'Win':
            data_by_team[team]['win'] = 1
        else:
            data_by_team[team]['win'] = 0

        # Unsure if necessary and if we can't just use team index but might as well
        if game_data['teams'][team]['teamId'] == 100:
            data_by_team[team]['side'] = 0
        else:
            data_by_team[team]['side'] = 1

        # Now add all the keys that we don't need to apply any processing to
        for json_key in unfiltered_keys[data_by_team[team]['side']]:
            # # Might be a worthwhile idea to look at role/lane and replace number with that?
            # # Reason being I'm unsure if it's consistent with position -> role
            # # Probably is in professional but _not_ in amateur?
            # # TODO sort this out
            # if json_key[2] == 'stats':
            #     base_attribute_key = ''
            #     pass
            for attribute in unfiltered_keys[data_by_team[team]['side']][json_key]:
                attribute_tuple = json_key + (attribute,)
                key_as_list = list(attribute_tuple)
                # Fix so that player numbers 5,6,7,8,9 return as 0,1,2,3,4
                if key_as_list[1] in range(5, 10):
                    key_as_list[1] -= 5
                global_attribute_key = '/'.join(map(str, key_as_list[1:]))
                try:
                    data_by_team[team][global_attribute_key] = access_json_by_tuple(
                        game_data,
                        attribute_tuple
                    )
                except KeyError:
                    data_by_team[team][global_attribute_key] = np.nan

        for team_stat in ['firstRiftHerald', 'dragonKills', 'inhibitorKills',
                          'firstBlood', 'firstInhibitor', 'towerKills',
                          'firstTower', 'firstDragon', 'baronKills',
                          'firstBaron'
                          ]:
            data_by_team[team][team_stat] = game_data['teams'][team][team_stat]

        data_by_team[team]['team_name'] = team_names[team]

    # Make dicts into dataframe and reorder columns a little
    summary_df = pd.DataFrame(data_by_team)
    cols_at_front = ['team_name', 'win', 'side']
    cols = list(summary_df)
    for col in cols_at_front:
        cols.remove(col)
    summary_df = summary_df[cols_at_front + cols]

    return summary_df


def game_data_simplified(game_realm, game_id, game_hash):
    """Processes stripped data and simplifies to a few parameters"""
    pass


def game_data_detailed(game_realm, game_id, game_hash):
    """Extracts features from the acs game data, really detailed"""
    pass

# Testing
if __name__ == '__main__':
    print(simple_game_row('TRLH3', '1001930081', 'f726d996f4363336'))
