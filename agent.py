import os
import re
import sys
import smtplib
import logging
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import feedparser
from dotenv import load_dotenv
from google import genai

load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Configurable settings via environment variables
LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", "24"))
MAX_ARTICLES = int(os.getenv("MAX_ARTICLES", "40"))
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; AINewsAgent/1.0; +https://github.com)"

FEEDS = [
    "https://techcrunch.com/category/artificial-intelligence/feed/",
    "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "https://www.wired.com/feed/tag/ai/latest/rss",
    "https://www.technologyreview.com/topic/artificial-intelligence/feed",
]


def clean_html_response(raw_text: str) -> str:
    """Safely strip markdown code fences without modifying valid HTML or links."""
    if not raw_text:
        return ""
    text = raw_text.strip()
    text = re.sub(r"^```(?:html)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def validate_environment() -> tuple[str, str, str, str]:
    """Validate required environment variables upfront before running pipeline."""
    gemini_key = os.getenv("GEMINI_API_KEY")
    gmail_user = os.getenv("GMAIL_USER")
    gmail_pass = os.getenv("GMAIL_APP_PASSWORD")
    recipient = os.getenv("RECIPIENT_EMAIL", gmail_user or "")

    missing = []
    if not gemini_key:
        missing.append("GEMINI_API_KEY")
    if not gmail_user:
        missing.append("GMAIL_USER")
    if not gmail_pass:
        missing.append("GMAIL_APP_PASSWORD")

    if missing:
        msg = f"Missing required environment variables: {', '.join(missing)}"
        logging.error(msg)
        raise ValueError(msg)

    return gemini_key, gmail_user, gmail_pass.replace(" ", ""), recipient


def parse_feed_entry(entry: dict, cutoff: datetime) -> dict | None:
    """Parse a single RSS/Atom entry with date and summary fallbacks."""
    time_struct = entry.get("published_parsed") or entry.get("updated_parsed")
    if not time_struct:
        return None

    try:
        published = datetime(*time_struct[:6], tzinfo=timezone.utc)
    except Exception:
        return None

    if published < cutoff:
        return None

    raw_summary = entry.get("summary") or entry.get("description", "")
    return {
        "title": entry.get("title", "Untitled").strip(),
        "link": entry.get("link", "").strip(),
        "summary": raw_summary[:300].strip(),
    }


def fetch_news() -> list[dict]:
    """Fetch news articles from configured feeds published within LOOKBACK_HOURS."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    news_items = []

    for feed_url in FEEDS:
        try:
            feed = feedparser.parse(feed_url, agent=DEFAULT_USER_AGENT)
            if getattr(feed, "status", None) and feed.status >= 400:
                logging.warning(f"Feed {feed_url} returned HTTP status {feed.status}")
                continue

            for entry in feed.entries:
                item = parse_feed_entry(entry, cutoff)
                if item:
                    news_items.append(item)
        except Exception as e:
            logging.error(f"Error fetching {feed_url}: {e}")

    return news_items


def summarize(articles: list[dict], gemini_key: str | None = None) -> str | None:
    """Generate an HTML news digest using the Gemini API."""
    if not articles:
        return None

    api_key = gemini_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.error("GEMINI_API_KEY is not configured.")
        return None

    lines = []
    for a in articles[:MAX_ARTICLES]:
        lines.append(f"- {a['title']} | {a['link']} | {a['summary']}")
    raw_text = "\n".join(lines)

    prompt = (
        "You are an AI news editor. From the articles below, write a daily "
        "AI news digest with the top 5 to 8 stories. Group them by theme, "
        "give each a 2-3 sentence summary, and include its link. "
        "Format the output as clean HTML (use <h3>, <p>, <a>). "
        "Return only the HTML, no markdown code fences.\n\n"
        + raw_text
    )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=MODEL, contents=prompt)
        return clean_html_response(response.text)
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return None


def create_email_message(html: str, sender: str, recipient: str, subject: str) -> MIMEMultipart:
    """Create a multipart email with both plain text and HTML alternatives."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient

    # Simple plain text fallback stripped of HTML tags
    plain_text = re.sub(r"<[^>]+>", "", html).strip()
    msg.attach(MIMEText(plain_text, "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))
    return msg


def send_email(html: str, user: str, password: str, recipient: str):
    """Send digest email via Gmail SMTP over SSL."""
    today = datetime.now(timezone.utc).strftime("%d %b %Y")
    subject = f"Daily AI News Digest - {today}"
    msg = create_email_message(html, sender=user, recipient=recipient, subject=subject)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
            server.login(user, password)
            server.send_message(msg)
        logging.info(f"Email successfully sent to {recipient}!")
    except Exception as e:
        logging.error(f"Email delivery error: {e}")
        raise


def main():
    try:
        gemini_key, user, password, recipient = validate_environment()
    except Exception:
        sys.exit(1)

    articles = fetch_news()
    logging.info(f"Retrieved {len(articles)} articles published within last {LOOKBACK_HOURS}h.")

    if not articles:
        logging.info("No new articles found. Exiting.")
        return

    digest = summarize(articles, gemini_key)
    if digest:
        try:
            send_email(digest, user, password, recipient)
        except Exception:
            sys.exit(1)
    else:
        logging.error("Failed to generate digest. Exiting with failure.")
        sys.exit(1)


if __name__ == "__main__":
    main()
