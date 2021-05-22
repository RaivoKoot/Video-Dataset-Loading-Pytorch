import json
import os

"""
This script takes in the raw json annotation files for 
SomethingSomethingV2 and turns them into annotation.txt files
that are compatible with this repository's dataloader VideoFrameDataset.

Running this script requires that you already have the SomethingSomethingV2
videos on disk as RGB frames, where each video has its own folder, containing
the RGB frames of that video. For this, you can use 
the script videos_to_frames.py.
"""

# the official Something Something V2 annotations file, either training or validation.
raw_annotations = 'something-something-v2-validation.json'
# the name of the output file
out_file = 'something-something-v2-validation-processed.txt'
# the official Something Something V2 label file, that specifies a mapping from TEXT_LABEL -> CLASS_ID
labels_file = 'something-something-v2-labels.json'

rgb_root = '../rgb/'

annotations = None

""" list containing the annotations for each sample """
with open(raw_annotations) as file:
    annotations = json.load(file)

""" dictionary to go from [text label] -> [integer label] """
with open(labels_file) as file:
    labels_to_ids_dict = json.load(file)


with open(out_file, 'w') as file:
    for sample in annotations:
        sample_id = sample['id']

        """ find out the number of frames for this sample """
        sample_rgb_directory = os.path.join(rgb_root,sample_id)
        num_frames = len(os.listdir(sample_rgb_directory)) - 1

        """ convert [text label] -> [integer label] """
        text_label = sample['template'].replace('[', '').replace(']', '')
        label_id = labels_to_ids_dict[text_label]

        """ write to processed file """
        annotation_string = "{} {} {} {}\n".format(sample_id, 0, num_frames, label_id)
        file.write(annotation_string)
