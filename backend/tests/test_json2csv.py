import unittest
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd
from core.json2csv import *


class TestJson2Csv(unittest.TestCase):
    @patch('core.json2csv.Path')
    @patch('builtins.open', new_callable=mock_open)
    @patch('core.json2csv.json.load')
    def test_json2csv_logic(self, mock_json_load, mock_file, mock_path):
        # Configuration des données factices
        cv_data = {
            "meta": {"cv_id": "cv123"},
            "age": 28,
            "total_experience_years": 4,
            "skills": ["Python", "SQL"],
            "languages": [
                {"language": "Français", "score": 5},
                {"language": "Japonais", "score": 3}  # Langue non mappée
            ],
            "experience_gaps_months": [{"duration_months": 2}]
        }
        mock_json_load.return_value = cv_data

        mock_file_path = MagicMock()
        mock_path.return_value.glob.return_value = [mock_file_path]

        df = json2csv("dummy_folder")

        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["cv_id"], "cv123")
        self.assertEqual(df.iloc[0]["lang_fr"], 5)
        self.assertEqual(df.iloc[0]["lang_en"], 0)
        self.assertEqual(df.iloc[0]["lang_other_score_sum"], 3)
        self.assertEqual(df.iloc[0]["total_gap_months"], 2)
        self.assertEqual(df.iloc[0]["nb_gaps"], 1)
        self.assertEqual(df.iloc[0]["skills"], "Python, SQL")


if __name__ == '__main__':
    unittest.main()