import unittest
from unittest.mock import patch, MagicMock
import os
import json
from core.analyzer import _get_groq_config, extract_cv
import core.analyzer as analyzer_module


class TestGetGroqConfig(unittest.TestCase):
    @patch.dict(os.environ, {"GROQ_API_KEY": "fake_key", "GROQ_MODEL": "fake_model", "GROQ_TEMPERATURE": "0.5"})
    def test_get_config_success(self):
        config = _get_groq_config()
        self.assertEqual(config["api_key"], "fake_key")
        self.assertEqual(config["model"], "fake_model")
        self.assertEqual(config["temperature"], 0.5)

    @patch.dict(os.environ, {}, clear=True)
    def test_get_config_missing_key(self):
        with self.assertRaises(EnvironmentError):
            _get_groq_config()


class TestExtractCv(unittest.TestCase):

    def _make_mock_response(self, content: str) -> MagicMock:
        """Construit un objet réponse Groq simulé."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = content
        return mock_response

    @patch("core.analyzer.time.sleep", return_value=None)
    @patch("core.analyzer.Groq")
    @patch.dict(os.environ, {"GROQ_API_KEY": "fake_key"})
    def test_success_on_first_attempt(self, mock_groq_class, mock_sleep):
        """Retourne le dict correct quand la réponse est un JSON valide."""
        expected = {"nom": "Dupont", "poste": "Développeur"}
        mock_client = mock_groq_class.return_value
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            json.dumps(expected)
        )

        result = extract_cv("texte du CV")

        self.assertEqual(result, expected)
        mock_client.chat.completions.create.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("core.analyzer.time.sleep", return_value=None)
    @patch("core.analyzer.Groq")
    @patch.dict(os.environ, {"GROQ_API_KEY": "fake_key"})
    def test_strips_markdown_fence(self, mock_groq_class, mock_sleep):
        """Supprime les balises ```json avant de parser le JSON."""
        expected = {"nom": "Martin"}
        mock_client = mock_groq_class.return_value
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            "```json\n" + json.dumps(expected) + "\n```"
        )

        result = extract_cv("texte du CV")
        self.assertEqual(result, expected)

    @patch("core.analyzer.time.sleep", return_value=None)
    @patch("core.analyzer.Groq")
    @patch.dict(os.environ, {"GROQ_API_KEY": "fake_key"})
    def test_retries_on_json_decode_error_then_succeeds(self, mock_groq_class, mock_sleep):
        """Réessaie après un JSONDecodeError et retourne le résultat au second appel."""
        expected = {"nom": "Leroy"}
        mock_client = mock_groq_class.return_value
        mock_client.chat.completions.create.side_effect = [
            self._make_mock_response("invalid json {{"),
            self._make_mock_response(json.dumps(expected)),
        ]

        result = extract_cv("texte du CV")

        self.assertEqual(result, expected)
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)
        mock_sleep.assert_called_once_with(analyzer_module.RETRY_DELAY_SEC)

    @patch("core.analyzer.time.sleep", return_value=None)
    @patch("core.analyzer.Groq")
    @patch.dict(os.environ, {"GROQ_API_KEY": "fake_key"})
    def test_retries_on_api_exception_then_succeeds(self, mock_groq_class, mock_sleep):
        """Réessaie après une erreur réseau/API et retourne le résultat au second appel."""
        expected = {"poste": "DevOps"}
        mock_client = mock_groq_class.return_value
        mock_client.chat.completions.create.side_effect = [
            Exception("Timeout réseau"),
            self._make_mock_response(json.dumps(expected)),
        ]

        result = extract_cv("texte du CV")

        self.assertEqual(result, expected)
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)

    @patch("core.analyzer.time.sleep", return_value=None)
    @patch("core.analyzer.Groq")
    @patch.dict(os.environ, {"GROQ_API_KEY": "fake_key"})
    def test_raises_last_exception_after_max_retries_skip_false(self, mock_groq_class, mock_sleep):
        """Propage la dernière exception après MAX_RETRIES échecs quand SKIP_ON_FAILURE=False."""
        mock_client = mock_groq_class.return_value
        mock_client.chat.completions.create.return_value = self._make_mock_response("invalid json")

        original_max = analyzer_module.MAX_RETRIES
        original_skip = analyzer_module.SKIP_ON_FAILURE
        try:
            analyzer_module.MAX_RETRIES = 2
            analyzer_module.SKIP_ON_FAILURE = False

            with self.assertRaises(json.JSONDecodeError):
                extract_cv("texte du CV")

            self.assertEqual(mock_client.chat.completions.create.call_count, 2)
        finally:
            analyzer_module.MAX_RETRIES = original_max
            analyzer_module.SKIP_ON_FAILURE = original_skip

    @patch("core.analyzer.time.sleep", return_value=None)
    @patch("core.analyzer.Groq")
    @patch.dict(os.environ, {"GROQ_API_KEY": "fake_key"})
    def test_raises_runtime_error_after_max_retries_skip_true(self, mock_groq_class, mock_sleep):
        """Lève un RuntimeError après MAX_RETRIES échecs quand SKIP_ON_FAILURE=True."""
        mock_client = mock_groq_class.return_value
        mock_client.chat.completions.create.return_value = self._make_mock_response("invalid json")

        original_max = analyzer_module.MAX_RETRIES
        original_skip = analyzer_module.SKIP_ON_FAILURE
        try:
            analyzer_module.MAX_RETRIES = 2
            analyzer_module.SKIP_ON_FAILURE = True

            with self.assertRaises(RuntimeError):
                extract_cv("texte du CV")
        finally:
            analyzer_module.MAX_RETRIES = original_max
            analyzer_module.SKIP_ON_FAILURE = original_skip

    @patch("core.analyzer.time.sleep", return_value=None)
    @patch("core.analyzer.Groq")
    @patch.dict(os.environ, {"GROQ_API_KEY": "fake_key"})
    def test_prompt_template_interpolation(self, mock_groq_class, mock_sleep):
        """Vérifie que le texte du CV est bien injecté dans le prompt envoyé à l'API."""
        expected = {"ok": True}
        mock_client = mock_groq_class.return_value
        mock_client.chat.completions.create.return_value = self._make_mock_response(
            json.dumps(expected)
        )

        extract_cv("MON_CV_UNIQUE")

        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        prompt_sent = messages[0]["content"]
        self.assertIn("MON_CV_UNIQUE", prompt_sent)


if __name__ == "__main__":
    unittest.main()