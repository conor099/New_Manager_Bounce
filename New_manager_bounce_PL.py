#%% Imports
import pandas as pd
from statsbombpy import sb
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# %% Load in a certain season from a certain league.

def SB_load_matches_from_season(league_name, season_name):
    """
    :param   league_name: Name of the league that you want data from (str) E.g. Premier League.
    :param   season_name: Name of the season that you want data from (str) E.g. 2003/2004.
    :return: matches_df:  Dataframe of all matches from selected season of selected league.
    """

    # The competitions that have Statsbomb data available.
    competitions_df = sb.competitions()

    # Extract SB competition id for desired competition.
    comp_id = competitions_df['competition_id'][
        (competitions_df['competition_name'] == league_name) & (competitions_df['season_name'] == season_name)].array[0]

    # Extract SB season id for desired season.
    seas_id = competitions_df['season_id'][
        (competitions_df['competition_name'] == league_name) & (competitions_df['season_name'] == season_name)].array[0]

    # Return the matches for the desired competition, ordered by matchdate.
    matches_df = sb.matches(competition_id=comp_id, season_id=seas_id).sort_values(by='match_date')

    return matches_df


#%% Extract PL matches for specific team in PL during 2015/16.

def extract_teams_matches_during_season(league_name, season_name, team_name):
    """
    :param   league_name:  Name of the league that you want data from (str) E.g. Premier League.
    :param   season_name:  Name of the season that you want data from (str) E.g. 2003/2004.
    :param   team_name:    Name of team whose matches you want to analyse. E.g. Arsenal.
    :return: team_matches: Dataframe containing matches for specified team.
    """
    # The competitions that have Statsbomb data available.
    all_matches = SB_load_matches_from_season(league_name=league_name, season_name=season_name)

    # Convert match_date column to be a datetime.
    all_matches["match_date"] = pd.to_datetime(all_matches["match_date"])

    # Extract matches by specified team.
    team_matches = all_matches[(all_matches["home_team"] == team_name) | (all_matches["away_team"] == team_name)]

    # Sort dataframe from first match to last match.
    team_matches = team_matches.sort_values(by="match_date")

    return team_matches


#%% Add a column to a teams' match dataframe to signify if a manager changed during a period.

def add_managerial_change_column(df, first_manager_hire_date, second_manager_hire_date="1753-01-01",
                                 number_hired_managers=1, number_of_matches_in_bounce=5):
    """
    :param   df:                          Dataframe for a specific team's matches.
    :param   first_manager_hire_date:     The date that a new manager was hired by a team. E.g. "2015-10-01". Swansea City and Aston Villa had 2 new managers hired, so a second hire date can be specified.
    :param   second_manager_hire_date:    The date that a second new manager was hired by a team (Only applies to Swansea City and Aston Villa).
    :param   number_hired_managers:       The number of managers a team hired during the season. Swansea City and Aston Villa were the only teams to hire 2 new managers during the season.
    :param   number_of_matches_in_bounce: Number of matches that you consider to be part of a "bounce" after a manager is hired.
    :return: df:                          Same dataframe returned but with one extra column to signify whether a game was part of a new bounce.
    """

    # If there was only new manager hired during the season:
    if number_hired_managers == 1:
        # Find the last date of the match where the new manager bounce still qualifies.
        last_match_date_of_bounce = df["match_date"][df["match_date"] > first_manager_hire_date].iloc[number_of_matches_in_bounce-1]

        # Add column to dataframe so that if the match is during a new manager bounce, it has value 1, and 0 otherwise.
        # Not bounce: Matches that occurred before manager change or greater than 5 games after the change.
        not_bounce_condition = (df["match_date"] < first_manager_hire_date) | (df["match_date"] > last_match_date_of_bounce)
        # Is bounce: The 5 matches that occurred after manager change.
        is_bounce_condition = (df["match_date"] >= first_manager_hire_date) & (df["match_date"] <= last_match_date_of_bounce)

        # Column added based on conditions.
        df["is_manager_bounce"] = np.select([not_bounce_condition, is_bounce_condition], [0, 1])

    # If number of new managers is 2. (This only applies to Swansea City and Aston Villa).
    elif number_hired_managers == 2:
        # Find the last date of the match where the new manager bounce still qualifies.
        last_match_date_of_first_bounce = df["match_date"][df["match_date"] >= first_manager_hire_date].iloc[number_of_matches_in_bounce - 1]
        last_match_date_of_second_bounce = df["match_date"][df["match_date"] >= second_manager_hire_date].iloc[number_of_matches_in_bounce - 1]

        # Add column to dataframe so that if the match is during a new manager bounce, it has value 1, and 0 otherwise.
        # Not bounce: Matches that occurred before manager change or greater than 5 games after the change.
        not_bounce_condition = (df["match_date"] < first_manager_hire_date) | \
                               ((df["match_date"] > last_match_date_of_first_bounce) & (df["match_date"] < second_manager_hire_date)) | \
                               (df["match_date"] > last_match_date_of_second_bounce)

        # Is bounce: The 5 matches that occurred after manager change.
        is_bounce_condition = (df["match_date"] >= first_manager_hire_date) & (df["match_date"] <= last_match_date_of_first_bounce) | \
                              (df["match_date"] >= second_manager_hire_date) & (df["match_date"] <= last_match_date_of_second_bounce)

        # Column added based on conditions.
        df["is_manager_bounce"] = np.select([not_bounce_condition, is_bounce_condition], [0, 1])

    return df


#%% Add a column to signify how many points a team had coming into a matchday.

def add_points_from_match_column(df, team_name):
    """
    :param   df:        Dataframe for a specific team's matches.
    :param   team_name: Name of specified team.
    :return: df:        Same dataframe as inputted, but with an extra column to show how many points a team won during a match.
    """
    # Initialise column to count the number of points the selected team won during a match.
    df["points_from_match"] = np.repeat(0, len(df))

    # Loop through each row of inputted dataframe. One row = One match.
    for index, row in df.iterrows():
        # Extract the home/away teams and scores for each match.
        home_team_name = row["home_team"]
        away_team_name = row["away_team"]
        home_team_score = row["home_score"]
        away_team_score = row["away_score"]
        # If specified team is playing at home:
        if home_team_name == team_name:
            # Award 3 points if specified team scored more than opposition.
            if home_team_score > away_team_score:
                df.loc[index, "points_from_match"] = 3
            # Award 1 point if specified team scored the same as the opposition.
            elif home_team_score == away_team_score:
                df.loc[index, "points_from_match"] = 1
            # Award 0 points if specified team scored less than opposition.
            else:
                df.loc[index, "points_from_match"] = 0

        # If specified team is playing away:
        elif away_team_name == team_name:
            # Award 3 points if specified team scored more than opposition.
            if away_team_score > home_team_score:
                df.loc[index, "points_from_match"] = 3
            # Award 1 point if specified team scored the same as the opposition.
            elif home_team_score == away_team_score:
                df.loc[index, "points_from_match"] = 1
            # Award 0 points if specified team scored less than opposition.
            else:
                df.loc[index, "points_from_match"] = 0

    return df


#%% Add both columns to a match dataframe for a specified team.

def add_manager_change_and_points_won_columns_to_df(team_name, first_manager_hire_date, second_manager_hire_date="1753-01-01",
                                                    number_hired_managers=1, number_of_matches_in_bounce=5):
    """
    :param   team_name:                   Name of specified team.
    :param   first_manager_hire_date:     The date that a new manager was hired by a team. E.g. "2015-10-01". Swansea City and Aston Villa had 2 new managers hired, so a second hire date can be specified.
    :param   second_manager_hire_date:    The date that a second new manager was hired by a team (Only applies to Swansea City and Aston Villa).
    :param   number_hired_managers:       The number of managers a team hired during the season. Swansea City and Aston VIlla were the only teams to hire 2 new managers during the season.
    :param   number_of_matches_in_bounce: Number of matches that you consider to be part of a "bounce" after a manager is hired.
    :return: match_df:                    Dataframe containing match data from SB, but with 2 new columns added signifying when a manager bounce occurred, and the team's points gained during a match.
    """
    # Extract match data for specified team from SB data.
    match_df = extract_teams_matches_during_season(league_name="Premier League", season_name="2015/2016",
                                                    team_name=team_name)

    # Add column to signify when a new manager bounce was occurring.
    match_df = add_managerial_change_column(df=match_df, first_manager_hire_date=first_manager_hire_date,
                                            second_manager_hire_date=second_manager_hire_date,
                                            number_hired_managers=number_hired_managers,
                                            number_of_matches_in_bounce=number_of_matches_in_bounce)

    # Add column to show how many points a team gained during each match.
    match_df = add_points_from_match_column(df=match_df, team_name=team_name)

    return match_df


#%% Find overall points per game difference between when teams do/don't have a new manager bounce.

def plot_overall_points_per_game_diff():
    """
    :return: Barchart of the difference in PPG in teams' bounce and non-bounce periods for teams with at least one managerial change in the Premier League 2015/16 season.
    """
    # Extract match dataframes for all teams that had at least one manager change during the 2015/16 season.
    # Also extract the points and games as tuples during/not during a new manager bounce.
    # Sunderland: Dick Advocaat (Left 2015-10-04) -> Sam Allardyce (Hired 2015-10-09).
    sunderland_matches = add_manager_change_and_points_won_columns_to_df(team_name="Sunderland",
                                                                         first_manager_hire_date="2015-10-09")
    sunderland_points_bounce = sunderland_matches.loc[sunderland_matches["is_manager_bounce"] == 1, "points_from_match"].tolist()
    sunderland_points_no_bounce = sunderland_matches.loc[sunderland_matches["is_manager_bounce"] == 0, "points_from_match"].tolist()

    # Liverpool: Brendan Rodgers (Left 2015-10-04) -> Jurgen Klopp (Hired 2015-10-08).
    liverpool_matches = add_manager_change_and_points_won_columns_to_df(team_name="Liverpool",
                                                                        first_manager_hire_date="2015-10-09")
    liverpool_points_bounce = liverpool_matches.loc[liverpool_matches["is_manager_bounce"] == 1, "points_from_match"].tolist()
    liverpool_points_no_bounce = liverpool_matches.loc[liverpool_matches["is_manager_bounce"] == 0, "points_from_match"].tolist()

    # Swansea City: 1st manager change- Garry Monk (Left 2015-12-09) -> Alan Curtis (Caretaker hired 2015-12-09)
    # 2nd manager change- Alan Curtis (Left 2016-01-18) -> Francesco Guidolin (Hired 2016-01-18).
    swansea_matches = add_manager_change_and_points_won_columns_to_df(team_name="Swansea City",
                                                                      first_manager_hire_date="2015-12-09",
                                                                      second_manager_hire_date="2016-01-18",
                                                                      number_hired_managers=2)
    swansea_points_bounce = swansea_matches.loc[swansea_matches["is_manager_bounce"] == 1, "points_from_match"].tolist()
    swansea_points_no_bounce = swansea_matches.loc[swansea_matches["is_manager_bounce"] == 0, "points_from_match"].tolist()

    # Aston VIlla: 1st manager change- Tim Sherwood (Left 2015-10-25) -> Remi Garde (Hired 2015-11-02).
    # 2nd manager change- Remi Garde (Left 2016-03-29) -> Eric Black (Hired 2016-03-29).
    av_matches = add_manager_change_and_points_won_columns_to_df(team_name="Aston Villa",
                                                                 first_manager_hire_date="2015-11-02",
                                                                 second_manager_hire_date="2016-03-29",
                                                                 number_hired_managers=2)
    av_points_bounce = av_matches.loc[av_matches["is_manager_bounce"] == 1, "points_from_match"].tolist()
    av_points_no_bounce = av_matches.loc[av_matches["is_manager_bounce"] == 0, "points_from_match"].tolist()

    # Chelsea: Jose Mourinho (Left 2015-12-17) -> Guus Hiddink (2015-12-20).
    chelsea_matches = add_manager_change_and_points_won_columns_to_df(team_name="Chelsea",
                                                                      first_manager_hire_date="2015-12-20")
    chelsea_points_bounce = chelsea_matches.loc[chelsea_matches["is_manager_bounce"] == 1, "points_from_match"].tolist()
    chelsea_points_no_bounce = chelsea_matches.loc[chelsea_matches["is_manager_bounce"] == 0, "points_from_match"].tolist()

    # Newcastle United: Steve McClaren (Left 2016-03-11) -> Rafael Benitez (2016-03-11).
    newcastle_matches = add_manager_change_and_points_won_columns_to_df(team_name="Newcastle United",
                                                                        first_manager_hire_date="2016-03-11")
    newcastle_points_bounce = newcastle_matches.loc[newcastle_matches["is_manager_bounce"] == 1, "points_from_match"].tolist()
    newcastle_points_no_bounce = newcastle_matches.loc[newcastle_matches["is_manager_bounce"] == 0, "points_from_match"].tolist()

    # Extract the points gained during all new manager bounces.
    all_bounce_matches = sunderland_points_bounce + liverpool_points_bounce + swansea_points_bounce + \
                         av_points_bounce + chelsea_points_bounce + newcastle_points_bounce

    # Extract the points gained outside of all new manager bounces.
    all_no_bounce_matches = sunderland_points_no_bounce + liverpool_points_no_bounce + swansea_points_no_bounce + \
                            av_points_no_bounce + chelsea_points_no_bounce + newcastle_points_no_bounce

    # Get the average points per game (PPG) during new manager bounce and not during.
    ppg_bounce = np.mean(all_bounce_matches)
    ppg_no_bounce = np.mean(all_no_bounce_matches)

    # Test significance in difference between the new manager bounce and non-bounce matches.
    t_statistic, p_value = ttest_ind(all_bounce_matches, all_no_bounce_matches, alternative="greater")
    alpha = 0.05  # Set significance level
    if p_value < alpha:
        print(f"P-value = {round(p_value, 4)}: The PPG during new manager bounces is statistically greater than the PPG outside of the new manager bounce.")
    else:
        print(f"P-value = {round(p_value, 4)}: There is no statistically significant difference in means between the two samples.")

    # Plotting the bar chart to compare PPG for the 2 different periods.
    fig, ax = plt.subplots()

    # Bar positions.
    bar_positions = np.arange(2)

    # Bar heights (PPG).
    bar_heights = [ppg_bounce, ppg_no_bounce]

    # Bar labels.
    bar_labels = ['Bounce', 'Non-Bounce']

    # Create bars with color. Green means that the PPG is higher for that period.
    if ppg_bounce > ppg_no_bounce:
        color_list = ['green', 'red']
    elif ppg_no_bounce > ppg_bounce:
        color_list = ['red', 'green']
    elif ppg_bounce == ppg_no_bounce:
        color_list = ['yellow', 'yellow']

    bars = ax.bar(bar_positions, bar_heights, align='center', color=color_list, alpha=0.7)

    # Add labels, title and subtitle.
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels)
    ax.set_ylabel("PPG")
    ax.set_title("New Manager Bounce vs Non-Bounce", y=1.05, fontsize=14)
    if p_value < 0.05:
        hypothesis_title = f"P-value {round(p_value, 4)} < 0.05: PPG statistically larger during new manager bounce."
    elif p_value >= 0.05:
        hypothesis_title = f"P-value {round(p_value, 4)} > 0.05: No statistical diff. between bounce and non-bounce."
    fig.suptitle(hypothesis_title, y=0.92, fontsize=10, fontweight='bold')

    # Add labels at the top of the bars showing the PPG.
    for bar, height in zip(bars, bar_heights):
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height:.3f}', ha='center', va='bottom', color='black', fontsize=10)

    # Add sample size to middle of bars.
    for bar, length in zip(bars, [len(all_bounce_matches), len(all_no_bounce_matches)]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f'N_matches={length}', ha='center', va='center', color='black', fontsize=10, fontstyle='oblique', fontweight='bold')

    # Show the barchart.
    plt.show()


plot_overall_points_per_game_diff()

#%% Find individual teams' points per game difference for teams that had a new manager.

def plot_teams_points_per_game_diff(team_name, first_manager_hire_date, second_manager_hire_date="1753-01-01",
                                    number_hired_managers=1, number_of_matches_in_bounce=5):
    """
    :param team_name:                   Name of specified team.
    :param first_manager_hire_date:     The date that a new manager was hired by a team. E.g. "2015-10-01". Swansea City and Aston Villa had 2 new managers hired, so a second hire date can be specified.
    :param second_manager_hire_date:    The date that a second new manager was hired by a team (Only applies to Swansea City and Aston Villa).
    :param number_hired_managers:       The number of managers a team hired during the season. Swansea City and Aston VIlla were the only teams to hire 2 new managers during the season.
    :param number_of_matches_in_bounce: Number of matches that you consider to be part of a "bounce" after a manager is hired.
    :return: Barchart of the difference in PPG in teams' bounce and non-bounce periods for specified team during the Premier League 2015/16 season.
    """
    # Extract match data frame for specified team during the Premier League 2015/16 season.
    matches = add_manager_change_and_points_won_columns_to_df(team_name=team_name,
                                                              first_manager_hire_date=first_manager_hire_date,
                                                              second_manager_hire_date=second_manager_hire_date,
                                                              number_hired_managers=number_hired_managers,
                                                              number_of_matches_in_bounce=number_of_matches_in_bounce)
    points_bounce = matches.loc[matches["is_manager_bounce"] == 1, "points_from_match"].tolist()
    points_non_bounce = matches.loc[matches["is_manager_bounce"] == 0, "points_from_match"].tolist()

    # Get the average points per game (PPG) during new manager bounce and not during.
    ppg_bounce = np.mean(points_bounce)
    ppg_no_bounce = np.mean(points_non_bounce)

    # Test significance in difference between the new manager bounce and non-bounce matches.
    t_statistic, p_value = ttest_ind(points_bounce, points_non_bounce, alternative="greater")
    alpha = 0.05  # Set significance level
    if p_value < alpha:
        print(f"P-value = {round(p_value, 4)}: The PPG during new manager bounces is statistically greater than the PPG outside of the new manager bounce.")
        hypothesis_title = f"P-value {round(p_value, 4)} < 0.05: PPG statistically larger during new manager bounce."
    else:
        print(f"P-value = {round(p_value, 4)}: There is no statistically significant difference in PPG between the two periods.")
        hypothesis_title = f"P-value {round(p_value, 4)} > 0.05: No statistical diff. between bounce and non-bounce."

    # Plotting the bar chart to compare PPG for the 2 different periods.
    fig, ax = plt.subplots()

    # Bar positions.
    bar_positions = np.arange(2)

    # Bar heights (PPG).
    bar_heights = [ppg_bounce, ppg_no_bounce]

    # Bar labels.
    bar_labels = ['Bounce', 'Non-Bounce']

    # Create bars with color. Green means that the PPG is higher for that period.
    if ppg_bounce > ppg_no_bounce:
        color_list = ['green', 'red']
    elif ppg_no_bounce > ppg_bounce:
        color_list = ['red', 'green']
    elif ppg_bounce == ppg_no_bounce:
        color_list = ['yellow', 'yellow']
    bars = ax.bar(bar_positions, bar_heights, align='center', color=color_list, alpha=0.7)

    # Add labels, title and subtitle.
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bar_labels)
    ax.set_ylabel("PPG")
    ax.set_title(f"{team_name} New Manager Bounce vs Non-Bounce", y=1.05, fontsize=14)
    fig.suptitle(hypothesis_title, y=0.92, fontsize=10, fontweight='bold')

    # Add labels at the top of the bars showing the PPG.
    for bar, height in zip(bars, bar_heights):
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height:.3f}', ha='center', va='bottom', color='black', fontsize=10)

    # Add sample size to middle of bars.
    for bar, length in zip(bars, [len(points_bounce), len(points_non_bounce)]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f'N_matches={length}', ha='center', va='center', color='black', fontsize=10, fontstyle='oblique',
                fontweight='bold')

    # Show the barchart.
    plt.show()


plot_teams_points_per_game_diff(team_name="Sunderland", first_manager_hire_date="2015-10-09")
plot_teams_points_per_game_diff(team_name="Liverpool", first_manager_hire_date="2015-10-09")
plot_teams_points_per_game_diff(team_name="Swansea City", first_manager_hire_date="2015-12-09",
                                second_manager_hire_date="2016-01-18", number_hired_managers=2)
plot_teams_points_per_game_diff(team_name="Aston Villa", first_manager_hire_date="2015-11-02",
                                second_manager_hire_date="2016-03-29", number_hired_managers=2)
plot_teams_points_per_game_diff(team_name="Chelsea", first_manager_hire_date="2015-12-20")
plot_teams_points_per_game_diff(team_name="Newcastle United", first_manager_hire_date="2016-03-11")


