# Using Something Something V2
This directory contains helpers to use the [Something Something V2](https://20bn.com/datasets/something-something) dataset with this 
repository's VideoFrameDataset dataloader.

### 1. Dataset Overview
When you download the Something Something V2 dataset, it comes in the following format:
- A `.webm` video file for every video
- A `.json` file for the training annotations and the validation annotations

To use VideoFrameDataset with Something Something V2, we need to
1. Create a folder for every `.webm` file that contains the RGB frames of that video.
2. Turn the `.json` file into an `annotations.txt` file, as described in the main README of this repository.

### 2. Processing
Doing (1) and (2) from above, is very easy if you use the python scripts provided in this directory.
- For (1), run the script `videos_to_frames.py` and make sure that you set the file paths
correctly inside of the script.
- For (2), run the script `raw_annotations_to_processed_annotations.py` and make sure that you
set the file paths correctly inside of the script. You also must have completed step (1),
before you are able to run this script. Run this script once for training and once for validation 
annotations.

NOTE: The processed training and validation files that step (2) outputs, are uploaded here as well.
You can directly use these and skip step (2). However, after completing step (1), you might have
to run (2) yourself, to create these two files yourself, in case there is some discrepancy between
the way `videos_to_frames.py` extracts RGB frames on my machine compared to on yours (this happened
to me once).

### 3. Done
That's it! You should then have a folder on your disk `RGB` that contains all videos in individual RGB
frames, and the two annotation files. This is all you need to use VideoFrameDataset and start training
on Something Something V2!
