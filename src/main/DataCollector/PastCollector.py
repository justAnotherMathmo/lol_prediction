# Standard Library
# import json
# 3rd Party
import pandas as pd
import numpy as np
import requests
# Repository Files
import _constants


def get_hashes_in_torunament(region, tournament):
    """Get hashes for all games that have been played in a given tournament"""
    if type(region) == int:
        request_url = '{0}/scheduleItems?leagueId={1}'.format(_constants.api_base_esports1,
                                                              region
                                                              )
    # elif type(region) == str:  # Untested and don't know how to match up leagues and slugs
    #     request_url = '{0}/leagues?slug={1}'.format(region)
    else:
        raise TypeError('region needs to be an int (or maybe a str)')

    # Fail fast error catching
    if attempt_req.status_code != 200:
        raise ConnectionError('Request Attempt to {} failed'.format(request_url))

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
    request_url = '{0}/highlanderMatchDetails?tournamentId={1}&matchId={2}'.format(_constants.api_base_esports2,
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
    request_url = '{0}/{1}/{2}{3}?gameHash={4}'.format(_constants.api_base_match_data,
                                                       game_realm,
                                                       game_id,
                                                       timeline_url_mod,
                                                       game_hash
                                                       )
    return requests.get(request_url).json()
