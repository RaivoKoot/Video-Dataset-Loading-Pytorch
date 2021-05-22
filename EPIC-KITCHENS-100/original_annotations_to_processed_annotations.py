import os
import pandas as pd

"""
This script converts the original EPIC-KITCHENS-100 annotation file 
and turns it into an annotation.txt file that is compatible 
with this repository's dataloader VideoFrameDataset.

Modify the two filepaths below and then run this script.
"""

if __name__ == '__main__':
    # filepath to where you have stored the original annotaiton file
    annotation_file = os.path.join(os.path.expanduser('~'), 'homedata', 'EPICKITCHENS', 'annotations', 'EPIC_100_train_subset.csv')

    # the output path and file name that you want to use
    out_file = os.path.join(os.path.expanduser('~'), 'data', 'EPICKITCHENS', 'annotations', 'EPIC_100_validation_new.txt')

    data = pd.read_csv(annotation_file, header=0)
    for i,row in enumerate(data):
        print(row)

    with open(out_file, 'a') as file:
        for index, row in data.iterrows():
            path = os.path.join(row['participant_id'], 'rgb_frames', row['video_id'])
            start_frame = row['start_frame']
            last_frame = row['stop_frame']
            verb_class = row['verb_class']
            noun_class = row['noun_class']

            file.write(f"{path} {start_frame} {last_frame} {verb_class} {noun_class}\n")
