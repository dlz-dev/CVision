import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
from core.preprocessor import *
from core.preprocessor import (_split_sections, parse_date,  extract_email,
                               extract_skills, extract_languages, extract_certifications,
                               extract_graduation_year, _geocode_with_fallback)


class TestSplitSections(unittest.TestCase):
    def test_split_sections(self):
        cv_text = "Name:\nJohn Doe\nEducation:\nMIT\nSkills:\nPython, Java"
        sections = _split_sections(cv_text)
        self.assertEqual(sections["Name"], "John Doe")
        self.assertEqual(sections["Education"], "MIT")
        self.assertEqual(sections["Skills"], "Python, Java")

class TestParseDate(unittest.TestCase):
    @patch('core.preprocessor.datetime')
    def test_parse_date_present(self, mock_datetime):
        mock_datetime.now.return_value = datetime(2026, 4, 10)
        res = parse_date("Present")
        self.assertEqual(res.year, 2026)

    def test_parse_date_valid(self):
        res = parse_date("2020-05")
        self.assertEqual(res.year, 2020)
        self.assertEqual(res.month, 5)

    def test_parse_date_invalid(self):
        res = parse_date("InvalidDate")
        self.assertIsNone(res)

class TestExtractEmail(unittest.TestCase):
    def test_extract_email_found(self):
        self.assertEqual(extract_email("Contact: test@email.com"), "test@email.com")

    def test_extract_email_not_found(self):
        self.assertIsNone(extract_email("No email here"))

class TestComputeAge(unittest.TestCase):
    @patch('core.preprocessor.datetime')
    def test_compute_age(self, mock_datetime):
        mock_datetime.now.return_value = datetime(2026, 4, 10)
        dob = datetime(1996, 4, 9)
        self.assertEqual(compute_age(dob), 30)

class TestGeocodeWithFallback(unittest.TestCase):
    @patch('core.preprocessor.geolocator.geocode')
    def test_geocode_exact_match(self, mock_geocode):
        mock_geocode.return_value = MagicMock(latitude=40.0, longitude=-70.0)
        res = _geocode_with_fallback("New York, USA")
        self.assertIsNotNone(res)
        mock_geocode.assert_called_once()

class TestComputeDistanceKm(unittest.TestCase):
    @patch('core.preprocessor._geocode_with_fallback')
    @patch('core.preprocessor.geodesic')
    def test_compute_distance_success(self, mock_geodesic, mock_fallback):
        mock_fallback.return_value = MagicMock(latitude=49.0, longitude=6.0)
        mock_geodesic.return_value.kilometers = 50.123
        dist = compute_distance_km("Metz, France")
        self.assertEqual(dist, 50.12)

class TestScoreLanguageLevel(unittest.TestCase):
    def test_score_valid(self):
        self.assertEqual(score_language_level("B2"), 4)

    def test_score_invalid(self):
        self.assertIsNone(score_language_level("XYZ"))

class TestScoreEducation(unittest.TestCase):
    def test_score_master(self):
        self.assertEqual(score_education("Master in Computer Science"), 4)

    def test_score_none(self):
        self.assertIsNone(score_education("Unknown Degree"))

class TestExtractSkills(unittest.TestCase):
    def test_extract_skills(self):
        text = "Programming: Python, C++\nSoft: Leadership"
        skills = extract_skills(text)
        self.assertListEqual(skills, ["Python", "C++", "Leadership"])

class TestExtractLanguages(unittest.TestCase):
    def test_extract_languages(self):
        text = "English - C1\nFrench - Native"
        langs = extract_languages(text)
        self.assertEqual(langs[0]["language"], "English")
        self.assertEqual(langs[0]["score"], 5)

class TestExtractCertifications(unittest.TestCase):
    def test_extract_certifications(self):
        text = "AWS Certified - 2021\nScrum Master"
        certs = extract_certifications(text)
        self.assertEqual(certs[0]["year"], 2021)
        self.assertIsNone(certs[1]["year"])

class TestExtractGraduationYear(unittest.TestCase):
    def test_extract_graduation_year(self):
        text = "BSc 2018\nMSc 2020"
        self.assertEqual(extract_graduation_year(text), 2020)

class TestCleanCvTextForLlm(unittest.TestCase):
    def test_clean_cv_text(self):
        text = "Education:\nMIT\nExperience:\nGoogle\nSkills:\nPython"
        cleaned = clean_cv_text_for_llm(text)
        self.assertIn("Education:\nMIT", cleaned)
        self.assertNotIn("Skills:\nPython", cleaned)

class TestComputeExperienceMetrics(unittest.TestCase):
    def test_compute_metrics(self):
        exps = [
            {"start": "2020-01", "end": "2021-01"},
            {"start": "2021-06", "end": "2022-06"}
        ]
        metrics = compute_experience_metrics(exps)
        self.assertEqual(metrics["total_experience_years"], 2.0)
        self.assertEqual(metrics["experience_gaps_months"][0]["duration_months"], 5)

class TestPreProcessCv(unittest.TestCase):
    @patch('core.preprocessor._split_sections')
    @patch('core.preprocessor.compute_distance_km')
    @patch('core.preprocessor.time.sleep')
    def test_pre_process(self, mock_sleep, mock_distance, mock_split):
        mock_split.return_value = {
            "Date of Birth": "1990-01-01",
            "Address": "Paris",
            "Education": "MSc 2015"
        }
        mock_distance.return_value = 300.0
        res = pre_process_cv("dummy text")
        self.assertIsNotNone(res["age"])
        self.assertEqual(res["distance_ville_haute_km"], 300.0)
        self.assertEqual(res["graduation_year"], 2015)

if __name__ == '__main__':
    unittest.main()