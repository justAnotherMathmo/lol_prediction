import json
import os
import sys
sys.path.insert(0, os.path.abspath('../../src/main'))
from DataCollector import PastCollector


# Just a function to make pretty printing json nicer
def json_print(data):
    print(json.dumps(data, indent=2))


print('Normal Data:')
detailed_data = PastCollector.get_data_from_hash('TRLH3',
                                                 '1001930081',
                                                 'f726d996f4363336'
                                                 )

# The keys are: 'gameId', 'seasonId', 'participants', 'gameType', 'gameMode',
#               'teams', 'gameCreation', 'mapId', 'queueId', 'platformId',
#               'participantIdentities', 'gameDuration', 'gameVersion'
# Keys seasonId, gameType, gameMode, gameCreation, mapId, queueId, gameVersion
#   are uninteresting from the perspective of pro-play (although we may
#   want to make sure that all games we look at have gameMode = CLASSIC
# The keys gameId and platformId are the '1001930081' and 'TRLH3' that have
#   to be fed in initially

# gameDuration is simple - how long did the game last?
json_print(detailed_data['gameDuration'])

# participantIdentities gives a map between player names and their id's (1-10)
json_print(detailed_data['participantIdentities'][1])

# teams gives a bunch of team data during that game - e.g. objectives taken,
#   were they the first team to get that objective (with bounties), etc.
#   Also gives the bans
json_print(detailed_data['teams'][1])

# participants gives more stats on that individual player (champion, kills,
#   damage taken, damage taken in different times of the game, etc. etc.
#   There's a fuck ton here. Have fun.
json_print(detailed_data['participants'][4])


print('Timeline Data:')
detailed_data_timeline = PastCollector.get_data_from_hash('TRLH3',
                                                          '1001930081',
                                                          'f726d996f4363336',
                                                          timeline=1
                                                          )
# Keys are frameInterval and frames - the latter contains the data we want
# Frames are a list as long as the (ceiling of the) number of minutes in the game

# Each frame has events, the types of which have been observed as:
#  ['WARD_PLACED', 'SKILL_LEVEL_UP', 'ITEM_PURCHASED', 'ITEM_DESTROYED',
#   'WARD_KILL', 'BUILDING_KILL', 'CHAMPION_KILL', 'ITEM_SOLD',
#   'ELITE_MONSTER_KILL', 'ITEM_UNDO']
# They also have timestamps and a bunch of other data
json_print(detailed_data_timeline['frames'][10]['events'][4])

# Each frame also has participantFrames, which is a dictionary with keys
#   for each champion in the game (keys are strings of integers 1 to 10)
# Each of these gives a little information on the champion at that time
#   (i.e. when the frame was taken - once a minute?)
json_print(detailed_data_timeline['frames'][10]['participantFrames']['1'])
json_print(detailed_data_timeline['frames'][11]['participantFrames']['1'])

# Finally we have timestamp, which is probably the start of the frame
# This indicates that frameInterval is measured in ms, as frames are probably
#   every minute, and go up by 60000 each time
print(detailed_data_timeline['frames'][0]['timestamp'],
      detailed_data_timeline['frames'][1]['timestamp'],
      sep='\n'
      )