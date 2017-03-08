import os
import sys
sys.path.insert(0, os.path.abspath('../../src/main'))
from DataCollector import PastCollector

detailed_data = PastCollector.get_data_from_hash('TRLH3',
                                                 '1001930081',
                                                 'f726d996f4363336'
                                                 )
# Keys are frameInterval and frames - the latter contains the data we want
# Frames are a list as long as the (ceiling of the) number of minutes in the game

# Each frame has events, the types of which have been observed as:
#  ['WARD_PLACED', 'SKILL_LEVEL_UP', 'ITEM_PURCHASED', 'ITEM_DESTROYED',
#   'WARD_KILL', 'BUILDING_KILL', 'CHAMPION_KILL', 'ITEM_SOLD',
#   'ELITE_MONSTER_KILL', 'ITEM_UNDO']
# They also have timestamps and a bunch of other data
print(detailed_data['frames'][10]['events'][4])

# Each frame also has participantFrames, which is a dictionary with keys
#   for each champion in the game (keys are strings of integers 1 to 10)
# Each of these gives a little information on the champion at that time
#   (i.e. when the frame was taken - once a minute?)
print(detailed_data['frames'][10]['participantFrames']['1'],
      detailed_data['frames'][11]['participantFrames']['1'],
      sep='\n'
      )

# Finally we have timestamp, which is probably the start of the frame
# This indicates that frameInterval is measured in ms, as frames are probably
#   every minute, and go up by 60000 each time
print(detailed_data['frames'][0]['timestamp'],
      detailed_data['frames'][1]['timestamp'],
      sep='\n'
      )