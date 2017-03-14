# Standard Library
# import time
# import json
# 3rd Party
import pandas as pd
import numpy as np
import requests
# Repository Files
import _constants


def json_requester(request_url, retries=200):
    attempt_req = requests.get(request_url, timeout=60)
    curr_attempts = 0
    while True:
        if attempt_req.status_code == 200:
            return attempt_req.json()
        elif attempt_req.status_code == 504 and curr_attempts < retries:
            curr_attempts += 1
            if curr_attempts in [retries//4, retries//2, (3*retries)//4]:
                print('Get request {} of {} to {} timed out (504), trying again'.format(curr_attempts,
                                                                                        retries,
                                                                                        request_url))
            continue
        else:
            raise ConnectionError('Request Attempt to {} failed with status code {}'.format(request_url,
                                                                                            attempt_req.status_code))


def get_hashes_in_tournament(region, tournaments=None):
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
    region_matches = json_requester(request_url)

    all_match_details = {}

    if type(tournaments) == str:
        tournaments = [tournaments]
    elif tournaments is None:
        tournaments = list(map(lambda x: x['id'], region_matches['highlanderTournaments']))
    for tournament_matches in region_matches['highlanderTournaments']:
        if tournament_matches['id'] not in tournaments:
            continue
        for bracket in tournament_matches['brackets']:
            for match in tournament_matches['brackets'][bracket]['matches']:
                if tournament_matches['brackets'][bracket]['matches'][match]['state'] == 'unresolved':
                    # This happens if the game hasn't been played
                    continue
                all_match_details[match] = []
                match_json = tournament_matches['brackets'][bracket]['matches'][match]['games']
                for game in match_json:
                    if 'gameId' in match_json[game]:
                        all_match_details[match] += [(game,
                                                      match_json[game]['gameId'],
                                                      match_json[game]['gameRealm'],
                                                      int(match_json[game]['name'][1:])  # Game number
                                                      )]
                all_match_details[match].sort(key=lambda x: x[3])  # Sort by game number

                # TODO see when this happens and if we can avoid
                if all_match_details[match] == []:
                    all_match_details.pop(match, None)
    return all_match_details


def get_hashes_from_series(tournament_id, match_id):
    """Get hash match history info from tournament and match ids"""
    request_url = '{0}/highlanderMatchDetails?tournamentId={1}&matchId={2}'.format(_constants.api_base_esports2,
                                                                                   tournament_id,
                                                                                   match_id
                                                                                   )
    highlander_match_details = json_requester(request_url)
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
    return json_requester(request_url)
