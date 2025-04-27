import pandas as pd
import os
from datetime import datetime

class FieldingDataCollector:
    def __init__(self):
        # Initialize the dataframe with required columns
        self.columns = [
            'Match No.', 'Innings', 'Team', 'Player Name', 'Ballcount', 'Position',
            'Short Description', 'Pick', 'Throw', 'Runs', 'Overcount', 'Venue'
        ]
        self.data = pd.DataFrame(columns=self.columns)
        
        # Define valid values for categorical fields
        self.valid_picks = ['clean pick', 'good throw', 'fumble', 'bad throw', 'catch', 'drop catch']
        self.valid_throws = ['run out', 'missed stumping', 'missed run out', 'stumping', 'direct hit', 'none']
        
        # Define common fielding positions
        self.fielding_positions = [
            'Wicketkeeper', 'Slip', 'Gully', 'Point', 'Cover', 'Mid-off', 'Mid-on',
            'Mid-wicket', 'Square leg', 'Fine leg', 'Third man', 'Long-off', 'Long-on',
            'Deep mid-wicket', 'Deep square leg', 'Deep point', 'Deep cover'
        ]
    
    def collect_match_info(self):
        """Collect basic match information"""
        print("\n===== CRICKET FIELDING ANALYSIS: MATCH INFORMATION =====\n")
        
        self.match_no = input("Enter Match Number: ")
        self.venue = input("Enter Venue: ")
        self.innings = input("Enter Innings (1 or 2): ")
        
        print("\nMatch information recorded successfully!")
        return True
    
    def collect_player_info(self):
        """Collect information about the players to analyze"""
        print("\n===== PLAYER INFORMATION =====\n")
        
        self.team = input("Enter Team Name: ")
        
        self.players = []
        num_players = 3  # We'll analyze 3 players as per the task
        
        print(f"\nEnter the names of {num_players} players to analyze:")
        for i in range(num_players):
            player_name = input(f"Player {i+1}: ")
            self.players.append(player_name)
        
        print("\nPlayer information recorded successfully!")
        return True
    
    def display_menu(self):
        """Display the main menu"""
        print("\n===== CRICKET FIELDING ANALYSIS: MAIN MENU =====\n")
        print("1. Record a fielding event")
        print("2. View current data")
        print("3. Export data to Excel")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        return choice
    
    def record_fielding_event(self):
        """Record a new fielding event"""
        print("\n===== RECORD FIELDING EVENT =====\n")
        
        # Select player
        print("Select player:")
        for i, player in enumerate(self.players):
            print(f"{i+1}. {player}")
        
        player_idx = int(input("Enter player number: ")) - 1
        if player_idx < 0 or player_idx >= len(self.players):
            print("Invalid player selection.")
            return False
        
        player_name = self.players[player_idx]
        
        # Enter ball information
        overcount = input("Enter over number: ")
        ballcount = input("Enter ball number in the over (1-6): ")
        
        # Select fielding position
        print("\nSelect fielding position:")
        for i, position in enumerate(self.fielding_positions):
            print(f"{i+1}. {position}")
        
        position_idx = int(input("Enter position number (or 0 to enter custom): "))
        if position_idx == 0:
            position = input("Enter custom position: ")
        elif position_idx < 1 or position_idx > len(self.fielding_positions):
            print("Invalid position selection.")
            return False
        else:
            position = self.fielding_positions[position_idx-1]
        
        # Enter description
        description = input("Enter short description of the fielding event: ")
        
        # Select pick type
        print("\nSelect pick type:")
        for i, pick in enumerate(self.valid_picks):
            print(f"{i+1}. {pick}")
        
        pick_idx = int(input("Enter pick type number: ")) - 1
        if pick_idx < 0 or pick_idx >= len(self.valid_picks):
            print("Invalid pick type selection.")
            return False
        
        pick = self.valid_picks[pick_idx]
        
        # Select throw type
        print("\nSelect throw type:")
        for i, throw in enumerate(self.valid_throws):
            print(f"{i+1}. {throw}")
        
        throw_idx = int(input("Enter throw type number: ")) - 1
        if throw_idx < 0 or throw_idx >= len(self.valid_throws):
            print("Invalid throw type selection.")
            return False
        
        throw = self.valid_throws[throw_idx]
        
        # Enter runs saved/conceded
        runs = input("Enter runs saved (+) or conceded (-): ")
        try:
            runs = int(runs)
        except ValueError:
            print("Invalid runs value. Please enter a number.")
            return False
        
        # Create a new row
        new_row = {
            'Match No.': self.match_no,
            'Innings': self.innings,
            'Team': self.team,
            'Player Name': player_name,
            'Ballcount': ballcount,
            'Position': position,
            'Short Description': description,
            'Pick': pick,
            'Throw': throw,
            'Runs': runs,
            'Overcount': overcount,
            'Venue': self.venue
        }
        
        # Add the row to the dataframe
        self.data = pd.concat([self.data, pd.DataFrame([new_row])], ignore_index=True)
        print(f"\nFielding event for {player_name} recorded successfully!")
        return True
    
    def view_current_data(self):
        """View the current collected data"""
        if self.data.empty:
            print("\nNo data has been recorded yet.")
            return
        
        print("\n===== CURRENT FIELDING DATA =====\n")
        
        # Group by player
        for player in self.players:
            player_data = self.data[self.data['Player Name'] == player]
            if not player_data.empty:
                print(f"\n{player} - {len(player_data)} events recorded:")
                for _, row in player_data.iterrows():
                    print(f"  Over {row['Overcount']}.{row['Ballcount']} at {row['Position']}: {row['Short Description']}")
                    print(f"    Pick: {row['Pick']}, Throw: {row['Throw']}, Runs: {row['Runs']}")
            else:
                print(f"\n{player} - No events recorded yet.")
    
    def export_to_excel(self):
        """Export the collected data to an Excel file"""
        if self.data.empty:
            print("\nNo data to export.")
            return False
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fielding_analysis_{timestamp}.xlsx"
        
        try:
            # Create a writer object
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                # Write the raw data to the first sheet
                self.data.to_excel(writer, sheet_name='Raw Data', index=False)
            
            print(f"\nData successfully exported to {filename}")
            return True
        except Exception as e:
            print(f"\nError exporting data: {e}")
            return False
    
    def run(self):
        """Run the data collection process"""
        print("\n===== CRICKET FIELDING ANALYSIS DATA COLLECTION =====\n")
        print("Welcome to the Cricket Fielding Analysis Data Collection tool!")
        print("This tool will help you collect and analyze fielding performance data for T20 matches.")
        
        # Collect match and player information
        self.collect_match_info()
        self.collect_player_info()
        
        # Main loop
        while True:
            choice = self.display_menu()
            
            if choice == '1':
                self.record_fielding_event()
            elif choice == '2':
                self.view_current_data()
            elif choice == '3':
                self.export_to_excel()
            elif choice == '4':
                print("\nThank you for using the Cricket Fielding Analysis Data Collection tool!")
                break
            else:
                print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    collector = FieldingDataCollector()
    collector.run()