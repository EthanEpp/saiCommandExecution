import random

def split_train_test(data_path, train_path, test_path, test_ratio=0.2):
    with open(data_path, 'r') as file:
        lines = file.readlines()
    
    # Split data by classes assuming each class block is separated by two newlines
    classes = ''.join(lines).split('\n\n')
    
    train_data = []
    test_data = []
    
    # Process each class data block
    for class_data in classes:
        if class_data.strip():
            # Split into individual lines
            entries = class_data.strip().split('\n')
            # Shuffle to randomize data selection for splitting
            random.shuffle(entries)
            # Calculate the number of test examples
            n_test = int(len(entries) * test_ratio)
            # Split data
            test_data.extend(entries[:n_test])
            train_data.extend(entries[n_test:])
    
    # Shuffle train and test data to randomize the order within each file
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    # Write train data to a file
    with open(train_path, 'w') as file:
        file.write('\n'.join(train_data) + '\n')
    
    # Write test data to a file
    with open(test_path, 'w') as file:
        file.write('\n'.join(test_data) + '\n')

# Example usage
split_train_test('command_phrases_eic_10_24_full.iob', 'command_phrases_eic_10_24_train.iob', 'command_phrases_eic_10_24_test.iob')

# ADD: Text Alvin and tell him I'll be home around 5 to move the furniture.