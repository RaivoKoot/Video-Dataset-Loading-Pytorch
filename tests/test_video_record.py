from video_dataset import VideoRecord
from unittest import TestCase
import tempfile
from pathlib import Path
import os


class GenericTestVideoRecord:

    FOURCC = None
    BYTES_PER_PIXEL = None
    TEST_FRAME_SIZE = (3, 2)

    def check_indexes(self, start_file, end_file, start_frame, end_frame, label, num_frames, frame_size):
        self.assertEqual(self.vr.start_file_idx, start_file)
        self.assertEqual(self.vr.end_file_idx, end_file)
        self.assertEqual(self.vr.start_frame, start_frame)
        self.assertEqual(self.vr.end_frame, end_frame)
        self.assertEqual(self.vr.label, label)
        self.assertEqual(self.vr.num_frames, num_frames)
        self.assertEqual(self.vr.frame_size, frame_size)

    def check_item(self, idx, relative_file_path, offset, frame_size):
        file_path, data_offset, frame_size = self.vr[idx]
        self.assertEqual(file_path, Path(self.directory.name) / relative_file_path)
        self.assertEqual(data_offset, offset)
        self.assertEqual(frame_size, frame_size)
    
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        os.mkdir(Path(self.directory.name) / 'test0')
        self.files = [open(Path(self.directory.name) / f"test0/testfile{i}.yuv", "wb") for i in range(10)]
        frame_width, frame_height = self.TEST_FRAME_SIZE
        for f in self.files:
            f.write(b'\x01'*int(self.BYTES_PER_PIXEL*frame_width*frame_height*4))
            f.close()

        frame_width, frame_height = self.TEST_FRAME_SIZE
        self.video_meta = {
            'fourcc': self.FOURCC,
            'width': frame_width,
            'height': frame_height
        }
        return super().setUp()
        
    def tearDown(self) -> None:
        self.directory.cleanup()
        return super().tearDown()

    def test_full_files(self):
        self.vr = VideoRecord('test0 1 2 3'.strip().split(' '), self.directory.name, "testfile{:1d}.yuv", self.video_meta)
        self.check_indexes(1, 2, 0, 3, 3, 8, 9)

        self.check_item(0,  "test0/testfile1.yuv", 0, (3,2))
        self.check_item(7,  "test0/testfile2.yuv", int(1.5*3*2*3), (3,2))
        self.check_item(-1, "test0/testfile2.yuv", int(1.5*3*2*3), (3,2))
        self.check_item(1,  "test0/testfile1.yuv", int(1.5*3*2*1), (3,2))
        self.check_item(3,  "test0/testfile1.yuv", int(1.5*3*2*3), (3,2))
        self.check_item(4,  "test0/testfile2.yuv", 0, (3,2))

    def test_partial_start_file(self):
        self.vr = VideoRecord('test0 1.2 2 4'.strip().split(' '), self.directory.name, "testfile{:1d}.yuv", self.video_meta)
        self.check_indexes(1, 2, 2, 3, 4, 6, 9)
    
    def test_partial_end_frame(self):
        self.vr = VideoRecord('test0 1 2.2 5'.strip().split(' '), self.directory.name, "testfile{:1d}.yuv", self.video_meta)
        self.check_indexes(1, 2, 0, 2, 5, 7, 9)

    def test_partial_start_and_end_frame(self):
        self.vr = VideoRecord('test0 1.1 2.2 5'.strip().split(' '), self.directory.name, "testfile{:1d}.yuv", self.video_meta)
        self.check_indexes(1, 2, 1, 2, 5, 6, 9)

        self.check_item(0,  "test0/testfile1.yuv", int(1.5*3*2*1), (3,2))
        self.check_item(5,  "test0/testfile2.yuv", int(1.5*3*2*2), (3,2))
        self.check_item(-1, "test0/testfile2.yuv", int(1.5*3*2*2), (3,2))
        self.check_item(1,  "test0/testfile1.yuv", int(1.5*3*2*2), (3,2))
        self.check_item(2,  "test0/testfile1.yuv", int(1.5*3*2*3), (3,2))
        self.check_item(3,  "test0/testfile2.yuv", 0, (3,2))

    def test_single_full_file(self):
        self.vr = VideoRecord('test0 1 1 5'.strip().split(' '), self.directory.name, "testfile{:1d}.yuv", self.video_meta)
        self.check_indexes(1, 1, 0, 3, 5, 4, 9)
    
    def test_single_partial_file(self):
        self.vr = VideoRecord('test0 1.1 1.2 5'.strip().split(' '), self.directory.name, "testfile{:1d}.yuv", self.video_meta)
        self.check_indexes(1, 1, 1, 2, 5, 2, 9)


class TestNV12VideoRecord(GenericTestVideoRecord, TestCase):
    FOURCC = 'nv12'
    BYTES_PER_PIXEL = 1.5