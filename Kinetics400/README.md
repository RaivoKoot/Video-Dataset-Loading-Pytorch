# Using Kineics 400
This directory contains helpers to use the [Kinetics 400](https://github.com/cvdfoundation/kinetics-dataset) dataset with this 
repository's VideoFrameDataset dataloader. Download it from [this URL](https://github.com/cvdfoundation/kinetics-dataset).

### 1. Dataset Overview
When you download the Kinetics 400 dataset, it comes in the following format:
- An `.mp4` video file for every video
- A `.csv` file for the training, validation, and testing annotations

To use VideoFrameDataset with Kinetics 400, we need to
1. Create a folder for every `.mp4` file that contains the RGB frames of that video.
2. Turn each `.csv` file into an `annotations.txt` file, as described in the main README of this repository.

### 2. Processing
Doing (1) and (2) from above, is very easy if you use the python scripts provided in this directory.
- For (1), make sure that all `.mp4` files (trainin, validation, and testing) are located in a single and the same
directory. Run the script `videos_to_frames.py` and make sure that you set the file paths
correctly inside of the script. This will probably take ~10 hours for Kinetics 400.
- For (2), run the script `process_annotation_file.py` once for each annotation `.csv` and make sure that you
set the file paths correctly inside of the script. You also must have completed step (1),
before you are able to run this script.

NOTE: The processed training, validation, and testing files that step (2) outputs, are uploaded here as well.
You can directly use these and skip step (2). However, after completing step (1), you might have
to run (2) yourself, to create these three annotation files yourself, in case there is some discrepancy between
the way `videos_to_frames.py` extracts RGB frames on my machine compared to on yours (This is very likely. I 
recommend running step 2 yourself).

### 3. Done
That's it! You should then have a folder on your disk `RGB` that contains all videos in individual RGB
frames, and the three annotation files. This is all you need to use VideoFrameDataset and start training
on Kinetics 400!
