# lol_prediction
Collaborative project to create a number of test models for gambling in League of Legends

## Tasks

### Data Collection
* Scrape raw data from pro matches
 * Turns out we don't need to scrape, there's an "official-unofficial" API https://gist.github.com/levi/e7e5e808ac0119e154ce#tourn
 * Turns out we can use API to also get information about what happened during the match (exact calls figured out by listening to the network calls on lolesports.com rather than using the above, which is old and outdated
 * Now need to map tournament-id to nicer names and and make the output data readable
* Store data locally on server
* Interface with riot API to pull data about amateur games
 * Create summary statistics and store those (likely too much data otherwise)

### Cleaning
* Further post-processing of data

### Analysis
* Develop tools to use data to have more useful predictors

### Prediction
* Predict outcomes of games in advance
* Predict other stats about games in advance