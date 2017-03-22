# 3rd Party imports
import pandas as pd
import numpy as np
# import sklearn.ensemble
# Repository Files
import _constants


def add_over_all_stats(df, stat_name, inplace=True):
    """Adds stat_name together for one player"""
    new_col = df[['{}/{}'.format(i, stat_name) for i in range(5)]].sum(axis=1)
    df['all/{}'.format(stat_name)] = new_col
    if not inplace:
        return df


def add_over_all_stats_new(df, stat_name, inplace=True):
    """Adds stat_name together for one player"""
    new_col_blue = df.ix[df.side == 1, ['{}/{}'.format(i, stat_name) for i in range(5)]].sum(axis=1)
    new_col_red = df.ix[df.side == 0, ['{}/{}'.format(i, stat_name) for i in range(5, 10)]].sum(axis=1)
    df.ix[df.side == 1, 'all/{}'.format(stat_name)] = new_col_blue
    df.ix[df.side == 0, 'all/{}'.format(stat_name)] = new_col_red
    if not inplace:
        return df


def simplify_dataframe(df, include_team_name=True):
    pivot_stats = ['kills', 'deaths', 'assists', 'goldEarned', 'totalDamageDealtToChampions',
                   'wardsPlaced', 'wardsKilled']
    pivot_timeline = []
    for desc in ['goldPerMinDeltas', 'csDiffPerMinDeltas', 'xpPerMinDeltas']:
        pivot_timeline += ['{}/{}'.format(desc, time) for time in ['0-10', '10-20', '20-30', '30-end']]

    team_stats = ['baronKills', 'dragonKills', 'firstBlood', 'firstTower']
    if include_team_name:
        team_stats.append('team_name')

    for new_stat in pivot_stats:
        add_over_all_stats_new(df, 'stats/{}'.format(new_stat))
    for new_stat in pivot_timeline:
        add_over_all_stats_new(df, 'timeline/{}'.format(new_stat))

    new_df = df[team_stats + list(map(lambda x: 'all/stats/{}'.format(x), pivot_stats))
                + list(map(lambda x: 'all/timeline/{}'.format(x), pivot_timeline))]

    return new_df


# def forest_training_data(df):
#     list_of_games = []
#     simple_df = simplify_dataframe(df, include_team_name=False)
#     for row_num in range(len(df)//2):
#         # Get agggreagte data for a match with both teams, flipping for games
#         team_blue = simple_df[simple_df.index == 2*row_num].squeeze()
#         team_red = simple_df[simple_df.index == 2*row_num+1].squeeze()
#         game_data1 = np.append(team_blue.as_matrix(), team_red.as_matrix())
#         game_data1 = np.append(game_data1, [0])
#         game_data2 = np.append(team_red.as_matrix(), team_blue.as_matrix())
#         game_data2 = np.append(game_data2, [1])
#         list_of_games += [game_data1, game_data2]
#
#     return list_of_games, list(df.win)


def team_data(df):
    simple_df = simplify_dataframe(df)
    gb = simple_df.groupby(['team_name']).mean()
    return gb


def forest_training_data(df):
    list_of_games = []
    team_df = team_data(df)
    for row_num in range(len(df)//2):
        # Get agggreagte data for a match with both teams, flipping for games
        blue_name = df[df.index == 2*row_num].squeeze()['team_name']
        red_name = df[df.index == 2*row_num+1].squeeze()['team_name']
        team_blue = team_df.loc[blue_name].squeeze()
        team_red = team_df.loc[red_name].squeeze()
        game_data1 = np.append(team_blue.as_matrix(), team_red.as_matrix())
        game_data1 = np.append(game_data1, [0])
        game_data2 = np.append(team_red.as_matrix(), team_blue.as_matrix())
        game_data2 = np.append(game_data2, [1])
        list_of_games += [game_data1, game_data2]

    return list_of_games, list(df.win)

if __name__ == '__main__':
    df = pd.read_csv(_constants.data_location + 'simple_game_data_leagueId=6.csv')
    training, winloss = forest_training_data(df)
    forest = sklearn.ensemble.RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=4)
    forest.fit(training, winloss)
    print(forest.oob_score_, forest.oob_score, forest.score(training, winloss))
