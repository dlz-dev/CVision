import unittest
from unittest.mock import patch, MagicMock
from core.loader import *

class TestLoadCv(unittest.TestCase):
    @patch('core.loader.Path')
    def test_load_cv_success(self, MockPath):
        mock_file = MagicMock()
        mock_file.is_file.return_value = True
        mock_file.suffix = ".txt"
        mock_file.read_text.return_value = "Contenu du CV"
        MockPath.return_value = mock_file

        result = load_cv("dummy.txt")
        self.assertEqual(result, "Contenu du CV")

    @patch('core.loader.Path')
    def test_load_cv_file_not_found(self, MockPath):
        mock_file = MagicMock()
        mock_file.is_file.return_value = False
        MockPath.return_value = mock_file

        with self.assertRaises(FileNotFoundError):
            load_cv("dummy.txt")

    @patch('core.loader.Path')
    def test_load_cv_invalid_format(self, MockPath):
        mock_file = MagicMock()
        mock_file.is_file.return_value = True
        mock_file.suffix = ".pdf"
        MockPath.return_value = mock_file

        with self.assertRaises(ValueError):
            load_cv("dummy.pdf")

    @patch('core.loader.Path')
    def test_load_cv_empty_file(self, MockPath):
        mock_file = MagicMock()
        mock_file.is_file.return_value = True
        mock_file.suffix = ".txt"
        mock_file.read_text.return_value = "   "
        MockPath.return_value = mock_file

        with self.assertRaises(ValueError):
            load_cv("dummy.txt")


class TestLoadCvsFromFolder(unittest.TestCase):
    @patch('core.loader.Path')
    def test_load_cvs_success(self, MockPath):
        mock_dir = MagicMock()
        mock_dir.is_dir.return_value = True

        mock_file1 = MagicMock()
        mock_file1.name = "cv1.txt"
        mock_file1.is_file.return_value = True
        mock_file1.suffix = ".txt"
        mock_file1.read_text.return_value = "Texte 1"

        mock_dir.glob.return_value = [mock_file1]
        MockPath.return_value = mock_dir

        # On simule le load_cv complet
        with patch('core.loader.load_cv', return_value="Texte 1"):
            result = load_cvs_from_folder("dummy_folder")
            self.assertEqual(result, {"cv1.txt": "Texte 1"})

    @patch('core.loader.Path')
    def test_load_cvs_not_a_directory(self, MockPath):
        mock_dir = MagicMock()
        mock_dir.is_dir.return_value = False
        MockPath.return_value = mock_dir

        with self.assertRaises(NotADirectoryError):
            load_cvs_from_folder("dummy_folder")

    @patch('core.loader.Path')
    def test_load_cvs_empty_directory(self, MockPath):
        mock_dir = MagicMock()
        mock_dir.is_dir.return_value = True
        mock_dir.glob.return_value = []
        MockPath.return_value = mock_dir

        with self.assertRaises(FileNotFoundError):
            load_cvs_from_folder("dummy_folder")


if __name__ == '__main__':
    unittest.main()