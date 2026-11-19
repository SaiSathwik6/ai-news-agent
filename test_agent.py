import os
import unittest
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart

import agent


class TestAgent(unittest.TestCase):
    def test_clean_html_response_strips_code_fences(self):
        """Code fences (```html and ```) should be stripped cleanly."""
        raw = "```html\n<h3>Test Headline</h3>\n<p>Some content</p>\n```"
        expected = "<h3>Test Headline</h3>\n<p>Some content</p>"
        self.assertEqual(agent.clean_html_response(raw), expected)

    def test_clean_html_response_preserves_html_word_and_links(self):
        """Valid text containing 'html' or links containing 'html' must NOT be corrupted."""
        raw = "```html\n<p>Learning html and parsing html5.</p>\n<a href=\"https://example.com/html-guide\">Link</a>\n```"
        cleaned = agent.clean_html_response(raw)
        self.assertIn("Learning html and parsing html5.", cleaned)
        self.assertIn("https://example.com/html-guide", cleaned)
        self.assertNotIn("```", cleaned)

    def test_clean_html_response_handles_empty(self):
        """None or empty string input returns empty string without error."""
        self.assertEqual(agent.clean_html_response(""), "")
        self.assertEqual(agent.clean_html_response(None), "")

    def test_validate_environment_missing_vars_raises_or_exits(self):
        """validate_environment should raise ValueError when required keys are absent."""
        old_env = os.environ.copy()
        try:
            os.environ.clear()
            with self.assertRaises((ValueError, SystemExit)):
                agent.validate_environment()
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    def test_validate_environment_success_and_whitespace_strip(self):
        """validate_environment returns tuple and strips whitespace from app password."""
        old_env = os.environ.copy()
        try:
            os.environ["GEMINI_API_KEY"] = "gemini_secret_123"
            os.environ["GMAIL_USER"] = "alice@gmail.com"
            os.environ["GMAIL_APP_PASSWORD"] = "abcd efgh ijkl mnop"
            os.environ["RECIPIENT_EMAIL"] = "bob@example.com"
            
            key, user, pwd, recipient = agent.validate_environment()
            self.assertEqual(key, "gemini_secret_123")
            self.assertEqual(user, "alice@gmail.com")
            self.assertEqual(pwd, "abcdefghijklmnop")
            self.assertEqual(recipient, "bob@example.com")
        finally:
            os.environ.clear()
            os.environ.update(old_env)

    def test_create_email_message_multipart(self):
        """Email message should be MIMEMultipart with both plain text and html parts."""
        html_content = "<h3>Daily AI</h3><p>Article 1</p>"
        msg = agent.create_email_message(
            html=html_content,
            sender="sender@gmail.com",
            recipient="recipient@gmail.com",
            subject="Test Subject"
        )
        self.assertIsInstance(msg, MIMEMultipart)
        self.assertEqual(msg["From"], "sender@gmail.com")
        self.assertEqual(msg["To"], "recipient@gmail.com")
        self.assertEqual(msg["Subject"], "Test Subject")
        
        payloads = msg.get_payload()
        self.assertEqual(len(payloads), 2)
        content_types = [part.get_content_type() for part in payloads]
        self.assertIn("text/plain", content_types)
        self.assertIn("text/html", content_types)

    def test_parse_feed_entry_supports_updated_parsed_and_description(self):
        """Feeds using updated_parsed instead of published_parsed, or description instead of summary, should be parsed."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=24)
        
        entry = {
            "title": "Atom Article",
            "link": "https://example.com/atom",
            "updated_parsed": now.timetuple(),
            "description": "Atom description text"
        }
        parsed = agent.parse_feed_entry(entry, cutoff)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["title"], "Atom Article")
        self.assertEqual(parsed["summary"], "Atom description text")

    def test_parse_feed_entry_filters_old_articles(self):
        """Articles published before cutoff should be filtered out."""
        old_time = datetime.now(timezone.utc) - timedelta(hours=48)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        entry = {
            "title": "Old Article",
            "link": "https://example.com/old",
            "published_parsed": old_time.timetuple(),
            "summary": "Old summary"
        }
        parsed = agent.parse_feed_entry(entry, cutoff)
        self.assertIsNone(parsed)


if __name__ == "__main__":
    unittest.main()
