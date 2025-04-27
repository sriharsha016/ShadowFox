# Cricket Fielding Analysis System

This system allows you to collect, analyze, and visualize fielding performance data for cricket players in T20 matches. It implements the performance metrics formula as specified in the task requirements.

## System Components

1. **Cricket Fielding Analysis** (`cricket_fielding_analysis.py`): The core analysis engine that processes fielding data and calculates performance scores.

2. **Fielding Data Collector** (`cricket_fielding_data_collector.py`): A command-line tool for collecting fielding data during a match.

## Performance Metrics Formula

The system calculates fielding performance using the following formula:

```
PS = (CP×WCP) + (GT×WGT) + (C×WC) + (DC×WDC) + (ST×WST) + (RO×WRO) + (MRO×WMRO) + (DH×WDH) + RS
```

Where:
- PS: Performance Score
- CP: Clean Picks
- GT: Good Throws
- C: Catches
- DC: Dropped Catches
- ST: Stumpings
- RO: Run Outs
- MRO: Missed Run Outs
- DH: Direct Hits
- RS: Runs Saved (positive for runs saved, negative for runs conceded)

And the weights are:
- WCP: Weight for Clean Picks (default: 1)
- WGT: Weight for Good Throws (default: 1)
- WC: Weight for Catches (default: 2)
- WDC: Weight for Dropped Catches (default: -2)
- WST: Weight for Stumpings (default: 2)
- WRO: Weight for Run Outs (default: 3)
- WMRO: Weight for Missed Run Outs (default: -1)
- WDH: Weight for Direct Hits (default: 2)

## Dataset Features

The system collects the following data for each fielding event:

- Match No.: Identifier for the match
- Innings: Which innings the data is being recorded for
- Team: The team in the field
- Player Name: The fielder involved in the action
- Ballcount: Sequence number of the ball in the over
- Position: Fielding position of the player at the time of the ball
- Short Description: Brief description of the fielding event
- Pick: Categorize the pick-up as clean pick, good throw, fumble, bad throw, catch, or drop catch
- Throw: Classify the throw as run out, missed stumping, missed run out, stumping, or direct hit
- Runs: Number of runs saved (+) or conceded (-) through the fielding effort
- Overcount: The over number in which the event occurred
- Venue: Location of the match

## How to Use

### Data Collection

1. Run the data collector script:
   ```
   python cricket_fielding_data_collector.py
   ```

2. Follow the prompts to enter match information, player details, and fielding events.

3. Export the collected data to Excel when finished.

### Data Analysis

1. If you've collected data using the collector tool, you can import the Excel file into the analysis script.

2. Alternatively, you can directly use the `CricketFieldingAnalysis` class in your code:
   ```python
   from cricket_fielding_analysis import CricketFieldingAnalysis
   
   # Initialize the analysis system
   analysis = CricketFieldingAnalysis()
   
   # Add fielding events
   analysis.add_fielding_event(
       match_no=1, innings=1, team='Team Name', player_name='Player Name',
       ballcount=1, position='Position', description='Description',
       pick='clean pick', throw='none', runs=1, overcount=1, venue='Venue'
   )
   
   # Generate player summary
   summary = analysis.generate_player_summary('Player Name')
   print(summary)
   
   # Export to Excel
   analysis.export_to_excel()
   
   # Create visualizations
   analysis.visualize_player_performance()
   ```

3. The example code in `cricket_fielding_analysis.py` demonstrates how to analyze fielding performance for three sample players.

## Output

The system generates the following outputs:

1. **Excel Spreadsheet**: Contains raw data, player summaries, and performance metrics explanation.

2. **Visualizations**: Creates bar charts and pie charts showing performance scores and action distributions for each player.

## Requirements

- Python 3.6 or higher
- pandas
- numpy
- matplotlib
- openpyxl

Install the required packages using:
```
pip install pandas numpy matplotlib openpyxl
```

## Example

The main script includes example data for three players (Rohit Sharma, MS Dhoni, and Ravindra Jadeja) to demonstrate how the system works. Run the script to see the analysis in action:

```
python cricket_fielding_analysis.py
```

This will generate an Excel file with the analysis and create visualizations in a `fielding_visualizations` directory.