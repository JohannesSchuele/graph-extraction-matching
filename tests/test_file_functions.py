import os
import unittest

from config import Config
from functions_files import remove_data_folders, make_folders


class TestFileFunctionsLocal(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        remove_data_folders('short_video')
        cls.config = Config('short_video.mp4')

    def test_make_folders(self):
        cl = self.config

        self.assertFalse(os.path.isdir(cl.raw_img_folder))
        self.assertFalse(os.path.isdir(cl.overlay_img_folder))
        make_folders(cl.list_of_folders)
        self.assertTrue(os.path.isdir(cl.raw_img_folder))
        self.assertTrue(os.path.isdir(cl.overlay_img_folder))


# class TestFileFunctionsNetwork(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls) -> None:
#         remove_data_folders('M:/ma/graph-training/data/short_video')
#         cls.config = Config('M:/ma/graph-training/data/short_video.mp4')
#
#     def test_make_folders(self):
#         cn = self.config
#
#         self.assertFalse(os.path.isdir(cn.raw_img_folder))
#         self.assertFalse(os.path.isdir(cn.overlay_img_folder))
#         make_folders(cn.list_of_folders)
#         self.assertTrue(os.path.isdir(cn.raw_img_folder))
#         self.assertTrue(os.path.isdir(cn.overlay_img_folder))
