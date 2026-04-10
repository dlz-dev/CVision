import unittest
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd
from core.features import *

class TestCvToFeatures(unittest.TestCase):
    def test_cv_to_features_full(self):
        cv = {
            "age": 30,
            "total_experience_years": 5.5,
            "years_since_graduation": 6,
            "skills": ["Python", "Java"],
            "languages": [{"language": "English"}],
            "certifications": [{"name": "AWS"}],
            "experiences": [{}, {}],
            "experience_gaps_months": [{}]
        }
        result = cv_to_features(cv)
        self.assertEqual(result["age"], 30)
        self.assertEqual(result["total_experience_years"], 5.5)
        self.assertEqual(result["nb_skills"], 2)
        self.assertEqual(result["nb_languages"], 1)
        self.assertEqual(result["nb_certifications"], 1)
        self.assertEqual(result["nb_experiences"], 2)
        self.assertEqual(result["nb_gaps"], 1)

    def test_cv_to_features_empty(self):
        result = cv_to_features({})
        self.assertEqual(result["age"], 0)
        self.assertEqual(result["nb_skills"], 0)


class TestLoadFeatures(unittest.TestCase):
    @patch('core.features.pd.read_csv')
    @patch('core.features.Path')
    @patch('builtins.open', new_callable=mock_open, read_data='{"age": 25}')
    def test_load_features(self, mock_file, mock_path, mock_read_csv):
        # Mock du fichier CSV de labels
        mock_read_csv.return_value = pd.DataFrame([
            {"filename": "cv1.txt", "label": 1}
        ])

        # Mock des fichiers JSON
        mock_json_file = MagicMock()
        mock_json_file.stem = "cv1"
        mock_path.return_value.glob.return_value = [mock_json_file]

        with patch('core.features.json.load', return_value={"age": 25}):
            X, y = load_features("dummy_folder", "dummy_labels.csv")

            self.assertEqual(len(X), 1)
            self.assertEqual(X.iloc[0]["age"], 25)
            self.assertEqual(y.iloc[0], 1)


if __name__ == '__main__':
    unittest.main()