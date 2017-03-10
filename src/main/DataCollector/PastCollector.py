# Standard Library
# import json
# 3rd Party
import pandas as pd
import numpy as np
import requests


# Constants - TODO move to config file to generalise
esports_api_base1 = 'http://api.lolesports.com/api/v1'
esports_api_base2 = 'http://api.lolesports.com/api/v2'
match_data_api_base = 'https://acs.leagueoflegends.com/v1/stats/game'
# Also found 'http://api.lolesports.com/api/v2/' while looking at ws TODO investigate usefulness
# Also maybe http://2015.na.lolesports.com/api/'


def get_tournament_hashes(region, tournament):
    """Get hashes for all games that have been played in a given tournament"""
    if type(region) == int:
        request_url = '{0}/scheduleItems?leagueId={1}'.format(esports_api_base1,
                                                              region
                                                              )
    # elif type(region) == str:  # Untested
    #     request_url = '{0}/leagues?slug={1}'.format(region)
    else:
        raise TypeError('region needs to be an int (or maybe a str)')

    all_match_details = {}

    region_matches = requests.get(request_url).json()
    tournament_matches = region_matches['highlanderTournaments'][tournament]
    for bracket in tournament_matches['brackets']:
        for match in tournament_matches['brackets'][bracket]['matches']:
            all_match_details[match] = []
            match_json = tournament_matches['brackets'][bracket]['matches'][match]['games']
            for game in match_json:
                if 'gameId' in match_json[game]:
                    all_match_details[match] += [(game,
                                                  match_json[game]['gameId'],
                                                  match_json[game]['gameRealm']
                                                  )]
    return all_match_details


def get_hashes_from_series(tournament_id, match_id):
    """Get hash match history info from tournament and match ids"""
    request_url = '{0}/highlanderMatchDetails?tournamentId={1}&matchId={2}'.format(esports_api_base2,
                                                                                   tournament_id,
                                                                                   match_id
                                                                                   )
    highlander_match_details = requests.get(request_url).json()
    return highlander_match_details['gameIdMappings']


# TODO refactor to use also in calling Riot's official API
def get_data_from_hash(game_realm, game_id, game_hash, timeline=0):
    """Given the acs information for a match, get all explicit data about what happened"""
    if timeline:
        timeline_url_mod = '/timeline'
    else:
        timeline_url_mod = ''
    request_url = '{0}/{1}/{2}{3}?gameHash={4}'.format(match_data_api_base,
                                                       game_realm,
                                                       game_id,
                                                       timeline_url_mod,
                                                       game_hash
                                                       )
    return requests.get(request_url).json()


def game_data_stripped(game_json):
    """Extracts features from the acs game data"""
    pass


def game_data_simplified(game_json):
    """Processes stripped data and simplifies to a few parameters"""
    pass


def game_data_detailed(game_json):
    """Extracts features from the acs game data, really detailed"""
    pass
