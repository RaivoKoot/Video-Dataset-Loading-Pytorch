import os
import os.path
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from pathlib import Path

KNOWN_FOURCC = {
    'NV12': {
        'format': 'yuv',
        'type': 'planar',
        'dtype': np.dtype('>i1'),
        'chroma_subsampling': (2, 0) }
}

class VideoRecord(object):
    """
    Helper class for class VideoFrameDataset. This class
    represents a video sample's metadata.

    Args:
        root_datapath: the system path to the root folder
                       of the videos.
        row: A list with four or more elements where 1) The first
             element is the path to the video sample's frames excluding
             the root_datapath prefix 2) The  second element is the starting frame id of the video
             3) The third element is the inclusive ending frame id of the video
             4) The fourth element is the label index.
             5) any following elements are labels in the case of multi-label classification
    """
    def __init__(self, row, root_datapath, file_tamplate, video_meta):
        self._data = row
        self._path = os.path.join(root_datapath, row[0])
        self.file_template = file_tamplate
        self.video_meta = video_meta
        self.files_frames = [self._get_frames_per_file(i) for i in range(self.start_file_idx, self.start_file_idx + self.num_files)]

    def _get_file_size(self, idx: int) -> int:
        return (Path(self.path) / Path(self.file_template.format(idx))).stat().st_size

    def _get_frames_per_file(self, file_idx: int) -> int:
        return self._get_file_size(file_idx) // self.frame_size

    def __len__(self):
        return self.num_frames

    def __getitem__(self, key):
        # based on https://stackoverflow.com/questions/675140/python-class-getitem-negative-index-handling
        # Do this until the index is greater than 0.
        while key < 0:
            # Index is a negative, so addition will subtract.
            key += len(self)
        if key >= len(self):
            raise IndexError

        absolute_key = key + self.start_frame
        tmp = 0
        for i, size in enumerate(self.files_frames):
            if tmp + size > absolute_key:
                break
            tmp += size
        offset = (key - tmp + self.start_frame) * self.frame_size
        file_idx = i + self.start_file_idx
        file_path = Path(self._path) / Path(self.file_template.format(file_idx))
        return file_path, offset, (self.video_meta['width'] , self.video_meta['height'])


    @property
    def frame_size(self):
        if self.video_meta['fourcc'].upper() in ['NV12']:
            #data_type = np.dtype('>i1')
            pixel_count = self.video_meta['width'] * self.video_meta['height'] 
            y_size = 1 * pixel_count
            fourcc = KNOWN_FOURCC[self.video_meta['fourcc'].upper()]
            chroma_factor = sum(fourcc['chroma_subsampling'])/8
            u_size = chroma_factor * pixel_count
            v_size = chroma_factor * pixel_count

            frame_size = int(y_size + u_size + v_size)
            return frame_size
        else:
            raise NotImplementedError

    @property
    def path(self):
        return self._path

    @property
    def num_files(self):
        return self.end_file_idx - self.start_file_idx + 1  # +1 because end frame is inclusive

    @property
    def num_frames(self):
        ret = sum(self.files_frames[:-1]) + self.end_frame - self.start_frame + 1
        return ret
    
    @property
    def start_file_idx(self):
        return int(self._data[1].split('.', 1)[0])

    @property
    def start_frame(self):
        if '.' in self._data[1]:
            return int(self._data[1].split('.', 1)[1])
        else:
            return 0

    @property
    def end_file_idx(self):
        return int(self._data[2].split('.', 1)[0])

    @property
    def end_frame(self):
        if '.' in self._data[2]:
            return int(self._data[2].split('.', 1)[1])
        else:
            return self.files_frames[-1] - 1

    @property
    def label(self):
        # just one label_id
        if len(self._data) == 4:
            return int(self._data[3])
        # sample associated with multiple labels
        else:
            return [int(label_id) for label_id in self._data[3:]]

    def __repr__(self) -> str:
        return str(type(self)) + f'{self._data}'

class VideoFrameDataset(torch.utils.data.Dataset):
    r"""
    A highly efficient and adaptable dataset class for videos.
    Instead of loading every frame of a video,
    loads x RGB frames of a video (sparse temporal sampling) and evenly
    chooses those frames from start to end of the video, returning
    a list of x PIL images or ``FRAMES x CHANNELS x HEIGHT x WIDTH``
    tensors where FRAMES=x if the ``ImglistToTensor()``
    transform is used.

    More specifically, the frame range [START_FRAME, END_FRAME] is divided into NUM_SEGMENTS
    segments and FRAMES_PER_SEGMENT consecutive frames are taken from each segment.

    Note:
        A demonstration of using this class can be seen
        in ``demo.py``
        https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch

    Note:
        This dataset broadly corresponds to the frame sampling technique
        introduced in ``Temporal Segment Networks`` at ECCV2016
        https://arxiv.org/abs/1608.00859.


    Note:
        This class relies on receiving video data in a structure where
        inside a ``ROOT_DATA`` folder, each video lies in its own folder,
        where each video folder contains the frames of the video as
        individual files or raw video file with a naming convention such as
        img_001.jpg ... img_059.jpg.
        For enumeration and annotations, this class expects to receive
        the path to a .txt file where each video sample has a row with four
        (or more in the case of multi-label, see README on Github)
        space separated values:
        ``VIDEO_FOLDER_PATH     START_FRAME      END_FRAME      LABEL_INDEX``.
        ``VIDEO_FOLDER_PATH`` is expected to be the path of a video folder
        excluding the ``ROOT_DATA`` prefix. For example, ``ROOT_DATA`` might
        be ``home\data\datasetxyz\videos\``, inside of which a ``VIDEO_FOLDER_PATH``
        might be ``jumping\0052\`` or ``sample1\`` or ``00053\``.
        In case of raw video files, START_FRAME and END_FRAME may denote whole 
        file or a specific frame in the file e.g. ``4.123`` means file no. 4, frame no. 123. 

    Args:
        root_path: The root path in which video folders lie.
                   this is ROOT_DATA from the description above.
        annotationfile_path: The .txt annotation file containing
                             one row per video sample as described above.
        num_segments: The number of segments the video should
                      be divided into to sample frames from.
        frames_per_segment: The number of frames that should
                            be loaded per segment. For each segment's
                            frame-range, a random start index or the
                            center is chosen, from which frames_per_segment
                            consecutive frames are loaded.
        imagefile_template: The image filename template that video frame files
                            have inside of their video folders as described above.
        raw_video_format: RAW video format name according to `FOURCC <https://www.fourcc.org/>`
        raw_video_size: a tuple containing width and height of a RAW frame
        transform: Transform pipeline that receives a list of PIL images/frames.
        random_shift: Whether the frames from each segment should be taken
                      consecutively starting from the center of the segment, or
                      consecutively starting from a random location inside the
                      segment range.
        test_mode: Whether this is a test dataset. If so, chooses
                   frames from segments with random_shift=False.

    """
    def __init__(self,
                 root_path: str,
                 annotationfile_path: str,
                 num_segments: int = 3,
                 frames_per_segment: int = 1,
                 imagefile_template: str='img_{:05d}.jpg',
                 raw_video_format: str = None,
                 raw_video_size = None,
                 transform = None,
                 random_shift: bool = True,
                 test_mode: bool = False):
        super(VideoFrameDataset, self).__init__()

        self.video_meta = {
            'fourcc': raw_video_format,
            'width': raw_video_size[0],
            'height': raw_video_size[1]
        }
        self.raw_video_format = raw_video_format
        self.raw_video_size = raw_video_size
        self.root_path = root_path
        self.annotationfile_path = annotationfile_path
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.imagefile_template = imagefile_template
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        self._parse_list()

    def _load_image(self, file_path, offset, size):
        if not self.raw_video_format:
            return [Image.open(file_path).convert('RGB')]
        else:
            if self.raw_video_format.upper() in KNOWN_FOURCC:
                fourcc = KNOWN_FOURCC[self.raw_video_format.upper()]
                data_type = fourcc['dtype']
                y_size = 1 * size[0] * size[1]
                chroma_plane_size = y_size // 4

                y_data = np.fromfile(file_path, data_type, y_size, offset=offset).reshape( (size[1], size[0]) )
                cb_data = np.fromfile(file_path, data_type, chroma_plane_size, offset=offset+y_size).reshape( (size[1] // 2, size[0] // 2 ) )
                cb_data = cb_data.repeat(2, axis=0)
                cb_data = cb_data.repeat(2, axis=1)
                cr_data = np.fromfile(file_path, data_type, chroma_plane_size, offset=offset+y_size+chroma_plane_size).reshape( (size[1]//2, size[0]//2 ) )
                cr_data = cr_data.repeat(2, axis=0)
                cr_data = cr_data.repeat(2, axis=1)

                ycbcr_data = np.dstack((y_data, cb_data, cr_data)) 

                return [Image.fromarray(ycbcr_data, mode='YCbCr')]
            else:                
                raise NotImplementedError

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' '), self.root_path, self.imagefile_template, self.video_meta) for x in open(self.annotationfile_path)]
        print(self.video_list)

    def _sample_indices(self, record):
        """
        For each segment, chooses an index from where frames
        are to be loaded from.

        Args:
            record: VideoRecord denoting a video sample.
        Returns:
            List of indices of where the frames of each
            segment are to be loaded from.
        """

        segment_duration = (record.num_frames - self.frames_per_segment + 1) // self.num_segments
        if segment_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), segment_duration) + np.random.randint(segment_duration, size=self.num_segments)

        # edge cases for when a video has approximately less than (num_frames*frames_per_segment) frames.
        # random sampling in that case, which will lead to repeated frames.
        else:
            offsets = np.sort(np.random.randint(record.num_frames, size=self.num_segments))

        return offsets

    def _get_val_indices(self, record):
        """
        For each segment, finds the center frame index.

        Args:
            record: VideoRecord denoting a video sample.
        Returns:
             List of indices of segment center frames.
        """
        if record.num_frames > self.num_segments + self.frames_per_segment - 1:
            offsets = self._get_test_indices(record)

        # edge case for when a video does not have enough frames
        else:
            offsets = np.sort(np.random.randint(record.num_frames, size=self.num_segments))

        return offsets

    def _get_test_indices(self, record):
        """
        For each segment, finds the center frame index.

        Args:
            record: VideoRecord denoting a video sample
        Returns:
            List of indices of segment center frames.
        """

        tick = (record.num_frames - self.frames_per_segment + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets

    def __getitem__(self, index):
        """
        For video with id index, loads self.NUM_SEGMENTS * self.FRAMES_PER_SEGMENT
        frames from evenly chosen locations.

        Args:
            index: Video sample index.
        Returns:
            a list of PIL images or the result
            of applying self.transform on this list if
            self.transform is not None.
        """
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self._get(record, segment_indices)

    def _get(self, record, indices):
        """
        Loads the frames of a video at the corresponding
        indices.

        Args:
            record: VideoRecord denoting a video sample.
            indices: Indices at which to load video frames from. (ndarray)
        Returns:
            1) A list of PIL images or the result
            of applying self.transform on this list if
            self.transform is not None.
            2) An integer denoting the video label.
        """
        images = list()
        for seg_ind in indices:
            frame_index = int(seg_ind)
            for i in range(self.frames_per_segment):
                file_path, data_offset, frame_size = record[i]
                seg_img = self._load_image(file_path, data_offset, frame_size)
                images.extend(seg_img)
                # image_indices.append(frame_index)
                # if frame_index < record.end_frame:
                #     frame_index += 1

        # sort images by index in case of edge cases where segments overlap each other because the overall
        # video is too short for num_segments*frames_per_segment indices.
        # _, images = (list(sorted_list) for sorted_list in zip(*sorted(zip(image_indices, images))))

        if self.transform is not None:
            images = self.transform(images)

        return images, record.label

    def __len__(self):
        return len(self.video_list)

class ImglistToTensor(torch.nn.Module):
    """
    Converts a list of PIL images in the range [0,255] to a torch.FloatTensor
    of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1].
    Can be used as first transform for ``VideoFrameDataset``.
    """
    def forward(self, img_list):
        """
        Converts each PIL image in a list to
        a torch Tensor and stacks them into
        a single tensor.

        Args:
            img_list: list of PIL images.
        Returns:
            tensor of size ``NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
        """
        return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list])
