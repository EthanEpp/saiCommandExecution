import os
import subprocess

# Define the arrays
positions = ['45_left', '45_right', 'center_35', 'center_71', 'center_106', 'center_141']
commands = ['hey_zelda', 'login', 'speed_test', 'start_timer', 'zoom_call', 'send_text', 'hide_clock']

# Base directory
base_dir = "max_background"

# Loop through positions and commands
for position in positions:
    print(f"Move to position: {position}")
    
    for command in commands:
        # Construct the output file path
        output_file = f"{base_dir}/{position}/{command}.wav"
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Construct the arecord command
        arecord_cmd = [
            "arecord", 
            "-D", "record_left_16k", 
            "-r", "16000", 
            "-c", "1", 
            "-f", "S16_LE", 
            "-d", "6", 
            output_file
        ]
        
        # Display the command and wait for the user to press Enter
        # print(f"Ready to record: {output_file}")
        print(f"Say a variation of the {command} command")
        input("Press Enter to start recording...")
        
        # Execute the command
        try:
            subprocess.run(arecord_cmd, check=True)
            print(f"Recording saved: {output_file} \n")
        except subprocess.CalledProcessError as e:
            print(f"Failed to record {output_file}: {e}")
