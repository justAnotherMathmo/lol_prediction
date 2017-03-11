# lol_prediction
Collaborative project to create a number of test models for gambling in League of Legends

## Tasks

### Data Collection (src/main/DataCollector)
* Have built structure to request data from pro matches, given either the game
* Can also get all games in a tournament by chaining the tournament lookup and series lookup
* To see data in its current form, look at (and run) doc/examples/proof_of_concept
* Also need to make the output data readable and output into some csvs (CURRENT WORKING TASK)
* Need to automate collation of data into csv's given a tournament (focus on LCK to being with)
 * Perhaps also make a script to request game data once a week automatically?
* Store data locally on server
* Interface with riot API to pull data about amateur games
 * Create summary statistics and store those (likely too much data otherwise)

### Cleaning (not-started)
* Further post-processing of data

### Analysis (not-started)
* Develop tools to use data to have more useful predictors

### Prediction/Statistics (not-started)
* Predict outcomes of games in advance
* Predict other stats about games in advance