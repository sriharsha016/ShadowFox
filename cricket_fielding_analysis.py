import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

class CricketFieldingAnalysis:
    def __init__(self):
        # Initialize the dataframe with required columns
        self.columns = [
            'Match No.', 'Innings', 'Team', 'Player Name', 'Ballcount', 'Position',
            'Short Description', 'Pick', 'Throw', 'Runs', 'Overcount', 'Venue'
        ]
        self.data = pd.DataFrame(columns=self.columns)
        
        # Define weights for performance metrics
        self.weights = {
            'WCP': 1,  # Weight for Clean Picks
            'WGT': 1,  # Weight for Good Throws
            'WC': 2,   # Weight for Catches
            'WDC': -2,  # Weight for Dropped Catches
            'WST': 2,   # Weight for Stumpings
            'WRO': 3,   # Weight for Run Outs
            'WMRO': -1, # Weight for Missed Run Outs
            'WDH': 2    # Weight for Direct Hits
        }
        
        # Define valid values for categorical fields
        self.valid_picks = ['clean pick', 'good throw', 'fumble', 'bad throw', 'catch', 'drop catch']
        self.valid_throws = ['run out', 'missed stumping', 'missed run out', 'stumping', 'direct hit', 'none']
        
    def add_fielding_event(self, match_no, innings, team, player_name, ballcount, position, 
                          description, pick, throw, runs, overcount, venue):
        """Add a new fielding event to the dataset"""
        # Validate inputs
        if pick.lower() not in self.valid_picks:
            print(f"Warning: '{pick}' is not a valid pick type. Valid options are: {self.valid_picks}")
            return False
            
        if throw.lower() not in self.valid_throws:
            print(f"Warning: '{throw}' is not a valid throw type. Valid options are: {self.valid_throws}")
            return False
        
        # Create a new row
        new_row = {
            'Match No.': match_no,
            'Innings': innings,
            'Team': team,
            'Player Name': player_name,
            'Ballcount': ballcount,
            'Position': position,
            'Short Description': description,
            'Pick': pick.lower(),
            'Throw': throw.lower(),
            'Runs': runs,
            'Overcount': overcount,
            'Venue': venue
        }
        
        # Add the row to the dataframe
        self.data = pd.concat([self.data, pd.DataFrame([new_row])], ignore_index=True)
        print(f"Added fielding event for {player_name}")
        return True
    
    def calculate_performance_score(self, player_name):
        """Calculate the performance score for a specific player"""
        # Filter data for the specified player
        player_data = self.data[self.data['Player Name'] == player_name]
        
        if player_data.empty:
            print(f"No data found for player: {player_name}")
            return 0
        
        # Count different fielding actions
        clean_picks = len(player_data[player_data['Pick'] == 'clean pick'])
        good_throws = len(player_data[player_data['Pick'] == 'good throw'])
        catches = len(player_data[player_data['Pick'] == 'catch'])
        dropped_catches = len(player_data[player_data['Pick'] == 'drop catch'])
        stumpings = len(player_data[player_data['Throw'] == 'stumping'])
        run_outs = len(player_data[player_data['Throw'] == 'run out'])
        missed_run_outs = len(player_data[player_data['Throw'] == 'missed run out'])
        direct_hits = len(player_data[player_data['Throw'] == 'direct hit'])
        runs_saved = player_data['Runs'].sum()
        
        # Calculate performance score using the formula
        # PS = (CP×WCP) + (GT×WGT) + (C×WC) + (DC×WDC) + (ST×WST) + (RO×WRO) + (MRO×WMRO) + (DH×WDH) + RS
        performance_score = (
            (clean_picks * self.weights['WCP']) +
            (good_throws * self.weights['WGT']) +
            (catches * self.weights['WC']) +
            (dropped_catches * self.weights['WDC']) +
            (stumpings * self.weights['WST']) +
            (run_outs * self.weights['WRO']) +
            (missed_run_outs * self.weights['WMRO']) +
            (direct_hits * self.weights['WDH']) +
            runs_saved
        )
        
        return performance_score
    
    def generate_player_summary(self, player_name):
        """Generate a summary of fielding performance for a player"""
        player_data = self.data[self.data['Player Name'] == player_name]
        
        if player_data.empty:
            print(f"No data found for player: {player_name}")
            return None
        
        # Count different fielding actions
        clean_picks = len(player_data[player_data['Pick'] == 'clean pick'])
        good_throws = len(player_data[player_data['Pick'] == 'good throw'])
        catches = len(player_data[player_data['Pick'] == 'catch'])
        dropped_catches = len(player_data[player_data['Pick'] == 'drop catch'])
        fumbles = len(player_data[player_data['Pick'] == 'fumble'])
        bad_throws = len(player_data[player_data['Pick'] == 'bad throw'])
        stumpings = len(player_data[player_data['Throw'] == 'stumping'])
        run_outs = len(player_data[player_data['Throw'] == 'run out'])
        missed_run_outs = len(player_data[player_data['Throw'] == 'missed run out'])
        missed_stumpings = len(player_data[player_data['Throw'] == 'missed stumping'])
        direct_hits = len(player_data[player_data['Throw'] == 'direct hit'])
        runs_saved = player_data['Runs'].sum()
        
        # Calculate performance score
        performance_score = self.calculate_performance_score(player_name)
        
        # Create summary dictionary
        summary = {
            'Player Name': player_name,
            'Team': player_data['Team'].iloc[0] if not player_data.empty else 'Unknown',
            'Total Actions': len(player_data),
            'Clean Picks': clean_picks,
            'Good Throws': good_throws,
            'Catches': catches,
            'Dropped Catches': dropped_catches,
            'Fumbles': fumbles,
            'Bad Throws': bad_throws,
            'Stumpings': stumpings,
            'Run Outs': run_outs,
            'Missed Run Outs': missed_run_outs,
            'Missed Stumpings': missed_stumpings,
            'Direct Hits': direct_hits,
            'Runs Saved': runs_saved,
            'Performance Score': performance_score
        }
        
        return summary
    
    def calculate_advanced_metrics(self, player_name):
        """Calculate advanced fielding metrics for a player"""
        player_data = self.data[self.data['Player Name'] == player_name]
        
        if player_data.empty:
            return {}
        
        # Basic counts
        clean_picks = len(player_data[player_data['Pick'] == 'clean pick'])
        good_throws = len(player_data[player_data['Pick'] == 'good throw'])
        catches = len(player_data[player_data['Pick'] == 'catch'])
        dropped_catches = len(player_data[player_data['Pick'] == 'drop catch'])
        fumbles = len(player_data[player_data['Pick'] == 'fumble'])
        bad_throws = len(player_data[player_data['Pick'] == 'bad throw'])
        stumpings = len(player_data[player_data['Throw'] == 'stumping'])
        run_outs = len(player_data[player_data['Throw'] == 'run out'])
        missed_run_outs = len(player_data[player_data['Throw'] == 'missed run out'])
        direct_hits = len(player_data[player_data['Throw'] == 'direct hit'])
        runs_saved = player_data['Runs'].sum()
        
        # Advanced metrics
        total_actions = len(player_data)
        total_opportunities = catches + dropped_catches  # Total catching opportunities
        total_run_out_opportunities = run_outs + missed_run_outs  # Total run out opportunities
        
        # Calculate efficiency ratios (avoid division by zero)
        catch_efficiency = (catches / total_opportunities * 100) if total_opportunities > 0 else 0
        run_out_efficiency = (run_outs / total_run_out_opportunities * 100) if total_run_out_opportunities > 0 else 0
        
        # Calculate impact score (weighted contribution to team's fielding)
        impact_score = (catches * 2 + run_outs * 3 + direct_hits * 2 + stumpings * 2 + runs_saved) - (dropped_catches * 2 + missed_run_outs * 1 + fumbles * 0.5 + bad_throws * 0.5)
        
        # Calculate fielding reliability index (higher is better)
        positive_actions = clean_picks + good_throws + catches + stumpings + run_outs + direct_hits
        negative_actions = dropped_catches + fumbles + bad_throws + missed_run_outs
        reliability_index = (positive_actions / (positive_actions + negative_actions) * 10) if (positive_actions + negative_actions) > 0 else 0
        
        # Calculate position-specific effectiveness
        positions = player_data['Position'].value_counts().to_dict()
        position_effectiveness = {}
        for position, count in positions.items():
            position_data = player_data[player_data['Position'] == position]
            position_runs_saved = position_data['Runs'].sum()
            position_effectiveness[position] = position_runs_saved / count if count > 0 else 0
        
        # Calculate match situation impact
        # For simplicity, we'll consider early overs (1-6), middle overs (7-15), and death overs (16-20)
        early_overs_data = player_data[player_data['Overcount'] <= 6]
        middle_overs_data = player_data[(player_data['Overcount'] > 6) & (player_data['Overcount'] <= 15)]
        death_overs_data = player_data[player_data['Overcount'] > 15]
        
        early_overs_impact = self.calculate_performance_score_from_data(early_overs_data)
        middle_overs_impact = self.calculate_performance_score_from_data(middle_overs_data)
        death_overs_impact = self.calculate_performance_score_from_data(death_overs_data)
        
        return {
            'Catch Efficiency (%)': round(catch_efficiency, 2),
            'Run Out Efficiency (%)': round(run_out_efficiency, 2),
            'Impact Score': round(impact_score, 2),
            'Reliability Index (0-10)': round(reliability_index, 2),
            'Position Effectiveness': position_effectiveness,
            'Early Overs Impact': round(early_overs_impact, 2),
            'Middle Overs Impact': round(middle_overs_impact, 2),
            'Death Overs Impact': round(death_overs_impact, 2)
        }
    
    def calculate_performance_score_from_data(self, data):
        """Calculate performance score from a subset of player data"""
        if data.empty:
            return 0
            
        # Count different fielding actions
        clean_picks = len(data[data['Pick'] == 'clean pick'])
        good_throws = len(data[data['Pick'] == 'good throw'])
        catches = len(data[data['Pick'] == 'catch'])
        dropped_catches = len(data[data['Pick'] == 'drop catch'])
        stumpings = len(data[data['Throw'] == 'stumping'])
        run_outs = len(data[data['Throw'] == 'run out'])
        missed_run_outs = len(data[data['Throw'] == 'missed run out'])
        direct_hits = len(data[data['Throw'] == 'direct hit'])
        runs_saved = data['Runs'].sum()
        
        # Calculate performance score using the formula
        performance_score = (
            (clean_picks * self.weights['WCP']) +
            (good_throws * self.weights['WGT']) +
            (catches * self.weights['WC']) +
            (dropped_catches * self.weights['WDC']) +
            (stumpings * self.weights['WST']) +
            (run_outs * self.weights['WRO']) +
            (missed_run_outs * self.weights['WMRO']) +
            (direct_hits * self.weights['WDH']) +
            runs_saved
        )
        
        return performance_score
        
    def export_to_excel(self, filename='cricket_fielding_analysis.xlsx'):
        """Export the collected data to an Excel file with enhanced formatting and analysis"""
        if self.data.empty:
            print("No data to export")
            return False
        
        try:
            # Create a writer object
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Write the raw data to the first sheet with improved formatting
                self.data.to_excel(writer, sheet_name='Raw Data', index=False)
                
                # Get unique players, teams, and venues
                players = self.data['Player Name'].unique()
                teams = self.data['Team'].unique()
                venues = self.data['Venue'].unique()
                
                # Create an enhanced player summary sheet
                summary_data = []
                advanced_metrics_data = []
                for player in players:
                    summary = self.generate_player_summary(player)
                    advanced_metrics = self.calculate_advanced_metrics(player)
                    
                    if summary:
                        summary_data.append(summary)
                        
                        # Prepare advanced metrics for export
                        advanced_metric_row = {
                            'Player Name': player,
                            'Team': summary['Team']
                        }
                        advanced_metric_row.update({k: v for k, v in advanced_metrics.items() 
                                                if not isinstance(v, dict)})
                        advanced_metrics_data.append(advanced_metric_row)
                
                if summary_data:
                    # Create enhanced player summary sheet
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Player Summary', index=False)
                    
                    # Apply conditional formatting to the Player Summary sheet
                    workbook = writer.book
                    worksheet = writer.sheets['Player Summary']
                    
                    # Format the Performance Score column with color scale
                    performance_score_col = summary_df.columns.get_loc('Performance Score') + 1  # +1 for Excel's 1-based indexing
                    last_row = len(summary_df) + 1  # +1 for header row
                    
                    # Create advanced metrics sheet
                    if advanced_metrics_data:
                        advanced_df = pd.DataFrame(advanced_metrics_data)
                        advanced_df.to_excel(writer, sheet_name='Advanced Metrics', index=False)
                    
                    # Create team comparison sheet
                    team_data = []
                    for team in teams:
                        team_players = self.data[self.data['Team'] == team]['Player Name'].unique()
                        team_score = sum(self.calculate_performance_score(player) for player in team_players)
                        team_actions = len(self.data[self.data['Team'] == team])
                        team_runs_saved = self.data[self.data['Team'] == team]['Runs'].sum()
                        
                        team_data.append({
                            'Team': team,
                            'Total Players': len(team_players),
                            'Total Actions': team_actions,
                            'Total Runs Saved': team_runs_saved,
                            'Team Performance Score': team_score,
                            'Average Score Per Player': team_score / len(team_players) if len(team_players) > 0 else 0
                        })
                    
                    if team_data:
                        team_df = pd.DataFrame(team_data)
                        team_df.to_excel(writer, sheet_name='Team Comparison', index=False)
                
                # Create a performance metrics sheet explaining the formula with enhanced formatting
                metrics_data = {
                    'Metric': ['Clean Picks (CP)', 'Good Throws (GT)', 'Catches (C)', 'Dropped Catches (DC)',
                              'Stumpings (ST)', 'Run Outs (RO)', 'Missed Run Outs (MRO)', 'Direct Hits (DH)', 'Runs Saved (RS)'],
                    'Weight': [self.weights['WCP'], self.weights['WGT'], self.weights['WC'], self.weights['WDC'],
                               self.weights['WST'], self.weights['WRO'], self.weights['WMRO'], self.weights['WDH'], 'N/A'],
                    'Description': [
                        'Clean collection of the ball',
                        'Accurate throw to the wicketkeeper/bowler',
                        'Successful catch taken',
                        'Opportunity to take a catch but missed',
                        'Successful stumping of a batsman',
                        'Successfully running out a batsman',
                        'Opportunity to run out a batsman but missed',
                        'Direct hit to the stumps resulting in a run out',
                        'Net runs saved through fielding efforts (positive) or conceded (negative)'
                    ],
                    'Impact Level': ['Medium', 'Medium', 'High', 'High Negative', 'High', 'Very High', 'Medium Negative', 'Very High', 'Variable']
                }
                metrics_df = pd.DataFrame(metrics_data)
                metrics_df.to_excel(writer, sheet_name='Performance Metrics', index=False)
                
                # Add enhanced formula explanation with examples
                formula_data = {
                    'Formula Component': [
                        'Basic Formula', 
                        'Example Calculation',
                        'Catch Efficiency',
                        'Run Out Efficiency',
                        'Impact Score',
                        'Reliability Index'
                    ],
                    'Formula': [
                        'PS = (CP×WCP) + (GT×WGT) + (C×WC) + (DC×WDC) + (ST×WST) + (RO×WRO) + (MRO×WMRO) + (DH×WDH) + RS',
                        'Player with 3 CP, 2 GT, 1 C, 0 DC, 0 ST, 1 RO, 0 MRO, 1 DH, 2 RS = (3×1) + (2×1) + (1×2) + (0×-2) + (0×2) + (1×3) + (0×-1) + (1×2) + 2 = 12',
                        'Catches / (Catches + Dropped Catches) × 100',
                        'Run Outs / (Run Outs + Missed Run Outs) × 100',
                        '(C×2 + RO×3 + DH×2 + ST×2 + RS) - (DC×2 + MRO×1 + Fumbles×0.5 + Bad Throws×0.5)',
                        '(Positive Actions / Total Actions) × 10'
                    ],
                    'Explanation': [
                        'Performance Score calculation formula where each action is multiplied by its corresponding weight',
                        'Numerical example showing how the formula is applied to a player\'s statistics',
                        'Measures the percentage of successful catches out of all catching opportunities',
                        'Measures the percentage of successful run outs out of all run out opportunities',
                        'Measures the overall impact of a player\'s fielding actions on the match outcome',
                        'Measures the reliability of a player\'s fielding on a scale of 0-10'
                    ]
                }
                formula_df = pd.DataFrame(formula_data)
                formula_df.to_excel(writer, sheet_name='Performance Formula', index=False)
                
                # Create a match situation analysis sheet
                match_situation_data = []
                for player in players:
                    player_data = self.data[self.data['Player Name'] == player]
                    if not player_data.empty:
                        # Analyze performance in different phases of the match
                        early_overs = player_data[player_data['Overcount'] <= 6]
                        middle_overs = player_data[(player_data['Overcount'] > 6) & (player_data['Overcount'] <= 15)]
                        death_overs = player_data[player_data['Overcount'] > 15]
                        
                        match_situation_data.append({
                            'Player Name': player,
                            'Team': player_data['Team'].iloc[0],
                            'Early Overs Actions': len(early_overs),
                            'Early Overs Runs Saved': early_overs['Runs'].sum() if not early_overs.empty else 0,
                            'Middle Overs Actions': len(middle_overs),
                            'Middle Overs Runs Saved': middle_overs['Runs'].sum() if not middle_overs.empty else 0,
                            'Death Overs Actions': len(death_overs),
                            'Death Overs Runs Saved': death_overs['Runs'].sum() if not death_overs.empty else 0,
                            'Most Active Phase': 'Early' if len(early_overs) >= max(len(middle_overs), len(death_overs)) else 
                                               ('Middle' if len(middle_overs) >= max(len(early_overs), len(death_overs)) else 'Death'),
                            'Most Effective Phase': 'Early' if early_overs['Runs'].sum() >= max(middle_overs['Runs'].sum(), death_overs['Runs'].sum()) else 
                                                  ('Middle' if middle_overs['Runs'].sum() >= max(early_overs['Runs'].sum(), death_overs['Runs'].sum()) else 'Death')
                        })
                
                if match_situation_data:
                    match_df = pd.DataFrame(match_situation_data)
                    match_df.to_excel(writer, sheet_name='Match Situation Analysis', index=False)
                
                # Create a venue analysis sheet
                venue_data = []
                for venue in venues:
                    venue_actions = len(self.data[self.data['Venue'] == venue])
                    venue_runs_saved = self.data[self.data['Venue'] == venue]['Runs'].sum()
                    
                    # Get top performer at this venue
                    venue_players = self.data[self.data['Venue'] == venue]['Player Name'].unique()
                    top_performer = ''
                    top_score = 0
                    
                    for player in venue_players:
                        player_venue_data = self.data[(self.data['Venue'] == venue) & (self.data['Player Name'] == player)]
                        score = self.calculate_performance_score_from_data(player_venue_data)
                        if score > top_score:
                            top_score = score
                            top_performer = player
                    
                    venue_data.append({
                        'Venue': venue,
                        'Total Actions': venue_actions,
                        'Total Runs Saved': venue_runs_saved,
                        'Top Performer': top_performer,
                        'Top Performer Score': top_score
                    })
                
                if venue_data:
                    venue_df = pd.DataFrame(venue_data)
                    venue_df.to_excel(writer, sheet_name='Venue Analysis', index=False)
            
            print(f"Enhanced data successfully exported to {filename}")
            return True
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False
            
            print(f"Data successfully exported to {filename}")
            return True
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False
    
    def visualize_player_performance(self, players=None):
        """Create enhanced visualizations for player performance with advanced metrics"""
        if self.data.empty:
            print("No data to visualize")
            return
        
        if players is None:
            players = self.data['Player Name'].unique()
        elif isinstance(players, str):
            players = [players]
        
        # Create a directory for visualizations if it doesn't exist
        viz_dir = 'fielding_visualizations'
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # Performance score comparison with enhanced styling
        scores = [self.calculate_performance_score(player) for player in players]
        
        plt.figure(figsize=(12, 7))
        bars = plt.bar(players, scores, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.title('Fielding Performance Score Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Players', fontsize=12)
        plt.ylabel('Performance Score', fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Add the score values on top of the bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.savefig(f'{viz_dir}/performance_scores.png', dpi=300)
        plt.close()
        
        # Create advanced metrics comparison
        advanced_metrics = {}
        for player in players:
            metrics = self.calculate_advanced_metrics(player)
            if metrics:
                advanced_metrics[player] = metrics
        
        if advanced_metrics:
            # Efficiency comparison
            plt.figure(figsize=(12, 7))
            
            x = np.arange(len(advanced_metrics))
            width = 0.35
            
            catch_efficiencies = [advanced_metrics[player].get('Catch Efficiency (%)', 0) for player in advanced_metrics]
            run_out_efficiencies = [advanced_metrics[player].get('Run Out Efficiency (%)', 0) for player in advanced_metrics]
            
            bar1 = plt.bar(x - width/2, catch_efficiencies, width, label='Catch Efficiency', color='lightblue', edgecolor='blue')
            bar2 = plt.bar(x + width/2, run_out_efficiencies, width, label='Run Out Efficiency', color='lightgreen', edgecolor='green')
            
            plt.xlabel('Players', fontsize=12)
            plt.ylabel('Efficiency (%)', fontsize=12)
            plt.title('Fielding Efficiency Comparison', fontsize=16, fontweight='bold')
            plt.xticks(x, list(advanced_metrics.keys()), rotation=45, fontsize=10)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add values on top of bars
            for i, v in enumerate(catch_efficiencies):
                plt.text(i - width/2, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
            
            for i, v in enumerate(run_out_efficiencies):
                plt.text(i + width/2, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(f'{viz_dir}/efficiency_comparison.png', dpi=300)
            plt.close()
            
            # Impact and Reliability comparison
            plt.figure(figsize=(12, 7))
            
            impact_scores = [advanced_metrics[player].get('Impact Score', 0) for player in advanced_metrics]
            reliability_indices = [advanced_metrics[player].get('Reliability Index (0-10)', 0) for player in advanced_metrics]
            
            ax1 = plt.subplot(111)
            bars = ax1.bar(x, impact_scores, width=0.6, color='coral', edgecolor='red', alpha=0.7, label='Impact Score')
            ax1.set_xlabel('Players', fontsize=12)
            ax1.set_ylabel('Impact Score', fontsize=12, color='red')
            ax1.tick_params(axis='y', labelcolor='red')
            ax1.set_xticks(x)
            ax1.set_xticklabels(list(advanced_metrics.keys()), rotation=45, fontsize=10)
            
            # Add values on top of bars
            for bar, score in zip(bars, impact_scores):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{score:.1f}', ha='center', va='bottom', fontsize=9)
            
            ax2 = ax1.twinx()
            ax2.plot(x, reliability_indices, 'o-', color='blue', linewidth=2, markersize=8, label='Reliability Index')
            ax2.set_ylabel('Reliability Index (0-10)', fontsize=12, color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            ax2.set_ylim(0, 10.5)  # Set y-limit for reliability index
            
            # Add values above points
            for i, v in enumerate(reliability_indices):
                ax2.text(i, v + 0.2, f'{v:.1f}', ha='center', va='bottom', fontsize=9, color='blue')
            
            # Add combined legend
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper right')
            
            plt.title('Impact Score and Reliability Index Comparison', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{viz_dir}/impact_reliability_comparison.png', dpi=300)
            plt.close()
            
            # Match situation analysis visualization
            plt.figure(figsize=(14, 8))
            
            # Create data for match situation analysis
            match_situation_data = {}
            for player in advanced_metrics:
                match_situation_data[player] = {
                    'Early Overs Impact': advanced_metrics[player].get('Early Overs Impact', 0),
                    'Middle Overs Impact': advanced_metrics[player].get('Middle Overs Impact', 0),
                    'Death Overs Impact': advanced_metrics[player].get('Death Overs Impact', 0)
                }
            
            # Set up the plot
            x = np.arange(len(match_situation_data))
            width = 0.25
            
            # Plot bars for each match phase
            early_impacts = [match_situation_data[player]['Early Overs Impact'] for player in match_situation_data]
            middle_impacts = [match_situation_data[player]['Middle Overs Impact'] for player in match_situation_data]
            death_impacts = [match_situation_data[player]['Death Overs Impact'] for player in match_situation_data]
            
            bar1 = plt.bar(x - width, early_impacts, width, label='Early Overs (1-6)', color='lightblue', edgecolor='blue')
            bar2 = plt.bar(x, middle_impacts, width, label='Middle Overs (7-15)', color='lightgreen', edgecolor='green')
            bar3 = plt.bar(x + width, death_impacts, width, label='Death Overs (16-20)', color='salmon', edgecolor='red')
            
            plt.xlabel('Players', fontsize=12)
            plt.ylabel('Impact Score', fontsize=12)
            plt.title('Match Situation Impact Analysis', fontsize=16, fontweight='bold')
            plt.xticks(x, list(match_situation_data.keys()), rotation=45, fontsize=10)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add values on top of bars
            for i, v in enumerate(early_impacts):
                if v != 0:
                    plt.text(i - width, v + 0.2, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
            
            for i, v in enumerate(middle_impacts):
                if v != 0:
                    plt.text(i, v + 0.2, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
            
            for i, v in enumerate(death_impacts):
                if v != 0:
                    plt.text(i + width, v + 0.2, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(f'{viz_dir}/match_situation_impact.png', dpi=300)
            plt.close()
        
        # Individual player breakdowns (enhanced versions)
        for player in players:
            summary = self.generate_player_summary(player)
            if not summary:
                continue
            
            # Extract relevant metrics for visualization
            metrics = ['Clean Picks', 'Good Throws', 'Catches', 'Dropped Catches', 
                      'Stumpings', 'Run Outs', 'Missed Run Outs', 'Direct Hits']
            values = [summary[metric] for metric in metrics]
            
            # Create an enhanced pie chart for action distribution
            plt.figure(figsize=(10, 8))
            colors = ['#66b3ff', '#99ff99', '#ffcc99', '#ff9999', '#c2c2f0', '#ffb3e6', '#c2f0c2', '#ff6666']
            explode = [0.05 if v > 0 else 0 for v in values]  # Only explode non-zero segments
            
            # Filter out zero values for better visualization
            non_zero_metrics = []
            non_zero_values = []
            non_zero_colors = []
            non_zero_explode = []
            
            for i, (metric, value) in enumerate(zip(metrics, values)):
                if value > 0:
                    non_zero_metrics.append(metric)
                    non_zero_values.append(value)
                    non_zero_colors.append(colors[i])
                    non_zero_explode.append(explode[i])
            
            if non_zero_values:  # Only create pie chart if there are non-zero values
                plt.pie(non_zero_values, labels=non_zero_metrics, autopct='%1.1f%%', startangle=90, 
                       shadow=True, explode=non_zero_explode, colors=non_zero_colors)
                plt.title(f'Fielding Action Distribution - {player}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f'{viz_dir}/{player.replace(" ", "_")}_action_distribution.png', dpi=300)
            plt.close()
            
            # Create an enhanced bar chart for the player's metrics
            plt.figure(figsize=(12, 6))
            bar_colors = ['#66b3ff' if v >= 0 else '#ff9999' for v in values]
            bars = plt.bar(metrics, values, color=bar_colors, edgecolor='navy', alpha=0.7)
            plt.title(f'Fielding Actions - {player}', fontsize=14, fontweight='bold')
            plt.xlabel('Action Type', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.xticks(rotation=45, fontsize=10)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                if height != 0:  # Only add text for non-zero bars
                    plt.text(bar.get_x() + bar.get_width()/2, height + 0.1, 
                            f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{viz_dir}/{player.replace(" ", "_")}_actions.png', dpi=300)
            plt.close()
            
            # Create radar chart for player's all-round fielding abilities
            if player in advanced_metrics:
                # Prepare data for radar chart
                categories = ['Catching', 'Run Outs', 'Reliability', 'Impact', 'Runs Saved']
                
                # Normalize values to 0-10 scale for radar chart
                catch_metric = advanced_metrics[player].get('Catch Efficiency (%)', 0) / 10  # Scale from 0-100% to 0-10
                runout_metric = advanced_metrics[player].get('Run Out Efficiency (%)', 0) / 10  # Scale from 0-100% to 0-10
                reliability = advanced_metrics[player].get('Reliability Index (0-10)', 0)  # Already on 0-10 scale
                impact = min(10, advanced_metrics[player].get('Impact Score', 0) / 2)  # Scale impact score to 0-10
                runs_saved = min(10, max(0, (summary['Runs Saved'] + 5) / 2))  # Adjust runs saved to 0-10 scale
                
                values = [catch_metric, runout_metric, reliability, impact, runs_saved]
                
                # Create radar chart
                angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                values += values[:1]  # Close the polygon
                angles += angles[:1]  # Close the polygon
                categories += categories[:1]  # Close the labels
                
                fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
                ax.plot(angles, values, 'o-', linewidth=2, color='blue')
                ax.fill(angles, values, alpha=0.25, color='blue')
                ax.set_thetagrids(np.degrees(angles), categories, fontsize=12)
                
                # Set y-ticks
                ax.set_yticks(np.arange(0, 11, 2))
                ax.set_yticklabels([f'{i}' for i in range(0, 11, 2)], fontsize=10)
                ax.set_ylim(0, 10)
                
                plt.title(f'Fielding Ability Radar - {player}', fontsize=14, fontweight='bold', y=1.1)
                plt.tight_layout()
                plt.savefig(f'{viz_dir}/{player.replace(" ", "_")}_radar.png', dpi=300)
                plt.close()
        
        print(f"Enhanced visualizations saved to {viz_dir} directory")

# Example usage
def main():
    # Initialize the analysis system
    analysis = CricketFieldingAnalysis()
    
    # Sample data collection for three players
    # You would typically collect this data by watching a match or from existing records
    
    # Player 1: Rohit Sharma
    analysis.add_fielding_event(
        match_no=1, innings=1, team='Mumbai Indians', player_name='Rohit Sharma',
        ballcount=2, position='Mid-off', description='Clean collection and throw',
        pick='clean pick', throw='none', runs=1, overcount=1, venue='Wankhede Stadium'
    )
    analysis.add_fielding_event(
        match_no=1, innings=1, team='Mumbai Indians', player_name='Rohit Sharma',
        ballcount=4, position='Slip', description='Caught the edge',
        pick='catch', throw='none', runs=0, overcount=3, venue='Wankhede Stadium'
    )
    analysis.add_fielding_event(
        match_no=1, innings=1, team='Mumbai Indians', player_name='Rohit Sharma',
        ballcount=1, position='Cover', description='Fumbled the ball',
        pick='fumble', throw='none', runs=-1, overcount=5, venue='Wankhede Stadium'
    )
    analysis.add_fielding_event(
        match_no=1, innings=1, team='Mumbai Indians', player_name='Rohit Sharma',
        ballcount=3, position='Long-on', description='Direct hit run out',
        pick='clean pick', throw='direct hit', runs=2, overcount=8, venue='Wankhede Stadium'
    )
    
    # Player 2: MS Dhoni
    analysis.add_fielding_event(
        match_no=1, innings=1, team='Chennai Super Kings', player_name='MS Dhoni',
        ballcount=2, position='Wicketkeeper', description='Quick stumping',
        pick='clean pick', throw='stumping', runs=1, overcount=2, venue='Wankhede Stadium'
    )
    analysis.add_fielding_event(
        match_no=1, innings=1, team='Chennai Super Kings', player_name='MS Dhoni',
        ballcount=5, position='Wicketkeeper', description='Caught behind',
        pick='catch', throw='none', runs=0, overcount=4, venue='Wankhede Stadium'
    )
    analysis.add_fielding_event(
        match_no=1, innings=1, team='Chennai Super Kings', player_name='MS Dhoni',
        ballcount=3, position='Wicketkeeper', description='Missed stumping chance',
        pick='clean pick', throw='missed stumping', runs=-1, overcount=7, venue='Wankhede Stadium'
    )
    analysis.add_fielding_event(
        match_no=1, innings=1, team='Chennai Super Kings', player_name='MS Dhoni',
        ballcount=6, position='Wicketkeeper', description='Run out from throw',
        pick='clean pick', throw='run out', runs=2, overcount=10, venue='Wankhede Stadium'
    )
    
    # Player 3: Ravindra Jadeja
    analysis.add_fielding_event(
        match_no=1, innings=1, team='Chennai Super Kings', player_name='Ravindra Jadeja',
        ballcount=1, position='Point', description='Diving stop',
        pick='clean pick', throw='none', runs=2, overcount=6, venue='Wankhede Stadium'
    )
    analysis.add_fielding_event(
        match_no=1, innings=1, team='Chennai Super Kings', player_name='Ravindra Jadeja',
        ballcount=4, position='Cover', description='Direct hit run out',
        pick='clean pick', throw='direct hit', runs=2, overcount=9, venue='Wankhede Stadium'
    )
    analysis.add_fielding_event(
        match_no=1, innings=1, team='Chennai Super Kings', player_name='Ravindra Jadeja',
        ballcount=2, position='Long-off', description='Good throw to keeper',
        pick='good throw', throw='none', runs=1, overcount=11, venue='Wankhede Stadium'
    )
    analysis.add_fielding_event(
        match_no=1, innings=1, team='Chennai Super Kings', player_name='Ravindra Jadeja',
        ballcount=5, position='Mid-wicket', description='Caught brilliantly',
        pick='catch', throw='none', runs=1, overcount=15, venue='Wankhede Stadium'
    )
    
    # Generate player summaries
    print("\nPlayer Summaries:")
    for player in ['Rohit Sharma', 'MS Dhoni', 'Ravindra Jadeja']:
        summary = analysis.generate_player_summary(player)
        print(f"\n{player}:")
        for key, value in summary.items():
            if key != 'Player Name':
                print(f"  {key}: {value}")
    
    # Export data to Excel
    analysis.export_to_excel()
    
    # Create visualizations
    analysis.visualize_player_performance()

if __name__ == "__main__":
    main()