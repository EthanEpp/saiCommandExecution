# Command Data Labeling Documentation

## Introduction
This document describes how command data is labeled and provides guidelines on how to create new data for commands. The labeling format is essential for training models to understand and process voice commands effectively.

## Data Format
Each labeled command consists of an input sentence and corresponding labels. The format is as follows:

- **BOS**: Beginning of Sentence
- **EOS**: End of Sentence
- **O**: No tag associated
- **B-<label_name>**: Beginning of a labeled entity
- **I-<label_name>**: Continuation of a labeled entity

The command phrase starts with **BOS** and ends with **EOS**. The **BOS** corresponds with the tag **O**, and the **EOS** corresponds with the command label (intent).

### Example
```
BOS can you resume the tourniquet EOS    O O O O O O tourniquet_resume
BOS hit play on the music EOS            O O O O O O start_music
BOS can you text jackie EOS              O O O O B-contact_name send_text
BOS open the system settings for me EOS  O O O B-app_name I-app_name O O open_app
BOS please set the tunes to 80 EOS       O O O O O O B-volume_increment set_music_volume
BOS i d like to hear some music now EOS  O O O O O O O O O start_music
BOS show dicom schedule EOS              O O O O schedule_today
BOS please bring benjamin lee into the call EOS O O O B-contact_name I-contact_name O O O add_to_call
BOS play the top-20 best chicane songs on deezer EOS O O O B-sort I-sort B-artist O O B-service PlayMusic
```

### Explanation
- The input command starts with **BOS** and ends with **EOS**.
- Words between **BOS** and **EOS** are separated by spaces.
- Corresponding labels are separated by tabs.
- **O** represents no tag.
- Labeled entities are tagged with **B-<label_name>** for the beginning and **I-<label_name>** for the continuation.
- The final label is the intent/command type, aligned with **EOS**.

## Current Commands and Labels
### Command Labels
There are 71 command labels, including but not limited to:
- `open_app`
- `start_video`
- `mute_mic`
- `log_out`
- `cancel_all_timers`
- `set_music_volume`
- `next_song`
- `timer_remaining`
- `room_smoke`
- `minimize_window`
- `unmute_mic`
- `duration_tourniquet`
- `hide_tourniquet`
- `decrease_volume`
- `room_humidity`
- `diagnostic_logs`
- `show_schedule`
- `start_stopwatch`
- `start_surgery`
- `stop_music`
- `stop_stopwatch`
- `increase_music_volume`
- `start_zoom_call`
- `start_music`
- `stop_video`
- `stop_speaking`
- `set_timer`
- `stop_timer`
- `end_call`
- `schedule_today`
- `google_search`
- `last_patient`
- `reset_stopwatch`
- `increase_volume`
- `patient_in`
- `log_in`
- `room_temp`
- `reset_timer`
- `tourniquet_reset`
- `send_text`
- `first_patient`
- `next_patient`
- `add_to_call`
- `clear_timer`
- `hide_app`
- `surgery_complete`
- `hide_all_timers`
- `patient_out`
- `set_volume`
- `tourniquet_down`
- `restart_timer`
- `decrease_music_volume`
- `show_tourniquet`
- `diagnostic_logs_usb`
- `previous_song`
- `reset_all_timers`
- `tourniquet_resume`
- `clear_stopwatch`
- `delete_text_content`
- `PlayMusic`
- `show_all_timers`
- `joint_commands`
- `timer_running_list`
- `enlarge_window`
- `restart_song`
- `GetWeather`
- `extend_tourniquet`
- `youtube_search`
- `tourniquet_up`
- `patient_out`
- `unknown`
- `open_ended`

Let me know if you need any further adjustments!

### Tag Labels
There are 56 tag labels, including but not limited to:
Can you also replace this tag labels list?
- `B-multi_command_splitter`
- `B-search_query`
- `B-album`
- `B-sort`
- `B-artist`
- `B-app_name`
- `B-country`
- `B-geographic_poi`
- `B-source_user`
- `B-playlist`
- `B-current_location`
- `B-state`
- `B-timer_length`
- `B-contact_name`
- `B-city`
- `B-spatial_relation`
- `B-time_range`
- `B-timer_name`
- `B-service`
- `B-timeRange`
- `B-music_item`
- `B-condition_description`
- `B-year`
- `B-text_content`
- `B-condition_temperature`
- `B-genre`
- `B-track`
- `B-volume_increment`
- `I-multi_command_splitter`
- `I-search_query`
- `I-album`
- `I-sort`
- `I-artist`
- `I-app_name`
- `I-country`
- `I-geographic_poi`
- `I-source_user`
- `I-playlist`
- `I-current_location`
- `I-state`
- `I-timer_length`
- `I-contact_name`
- `I-city`
- `I-spatial_relation`
- `I-time_range`
- `I-timer_name`
- `I-service`
- `I-timeRange`
- `I-music_item`
- `I-condition_description`
- `I-year`
- `I-text_content`
- `I-condition_temperature`
- `I-genre`
- `I-track`
- `I-volume_increment`


## Creating New Data
To create a new command:
1. Follow the provided format.
2. Use automation to generate non-labeled data for a specific command type.
3. Manually verify label placement.
4. Run the data through a tagging script to assign appropriate labels.
5. Verify the output at each step.

### Automation Guidelines
- Use a language model to generate a list of commands.
- Ensure the commands have placeholder labels (e.g., using apostrophes).
- Manually review the generated commands to ensure accuracy.
- Tag the commands with the provided script.
- Verify the final labeled data to ensure correctness.

### Important Note
Fully automating data generation and labeling using language models has shown poor performance. Manual verification is crucial at every step to maintain data quality.
