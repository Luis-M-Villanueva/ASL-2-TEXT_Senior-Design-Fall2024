import json
import os
import shutil

# Load the JSON file
with open('temp code and helpers\\WLASL_v0.3.json', 'r') as f:
    data = json.load(f)

# Ask the user for the number of glosses
num_glosses = int(input("Enter the number of glosses: "))

# Ask the user for each gloss ID
gloss_ids = []
for i in range(num_glosses):
    gloss_id = input(f"Enter gloss ID {i+1}: ")
    gloss_ids.append(gloss_id)

# Create the output directory if it doesn't exist
output_dir = 'D:\\senior_design\\temp_test_vids'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through each gloss ID
for gloss_id in gloss_ids:
    
    # Create a directory for the chosen gloss
    gloss_dir = os.path.join(output_dir, gloss_id)
    if not os.path.exists(gloss_dir):
        os.makedirs(gloss_dir)
    
    # Find the video IDs for the chosen gloss
    video_ids = []
    found = False
    for item in data:
        if item['gloss'] == gloss_id:
            found = True
            for instance in item['instances']:
                video_ids.append(instance['video_id'])

    # If the gloss ID doesn't exist, print a message and skip it
    if not found:
        print(f"Gloss ID '{gloss_id}' does not exist. Skipping...")
        continue
    
    # Copy videos to the output directory
    for video_id in video_ids:
        video_path = os.path.join('D:\\senior_design\\processed_vids', str(video_id) + '.mp4')
        if os.path.exists(video_path):
            shutil.copy(video_path, gloss_dir)
        else:
            print(f"Video {video_id} does not exist.")