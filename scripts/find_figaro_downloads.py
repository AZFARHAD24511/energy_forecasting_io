
from html.parser import HTMLParser
from urllib.request import Request, urlopen
from urllib.parse import urljoin
from pathlib import Path

PROJECT = Path(r"D:\projects\energy_forecasting_io")
OUT = PROJECT / "data" / "raw" / "figaro"
OUT.mkdir(parents=True, exist_ok=True)

EUROSTAT_URL = (
    "https://ec.europa.eu/eurostat/en/web/"
    "esa-supply-use-input-tables/database"
)

class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        attrs = dict(attrs)
        href = attrs.get("href", "")
        text = attrs.get("title", "")
        if href:
            self.links.append((href, text))

def fetch(url):
    req = Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urlopen(req, timeout=40) as response:
        return response.read().decode("utf-8", errors="replace")

print("Reading official Eurostat page...")
html = fetch(EUROSTAT_URL)

parser = LinkParser()
parser.feed(html)

circabc_links = []
for href, label in parser.links:
    full_url = urljoin(EUROSTAT_URL, href)
    if "circabc.europa.eu" in full_url.lower():
        circabc_links.append((full_url, label))

circabc_links = list(dict.fromkeys(circabc_links))

print("\nOfficial CIRCABC links found:")
for url, label in circabc_links:
    print(f"- {url}")
    if label:
        print(f"  Label: {label}")

if not circabc_links:
    print(
        "\nNo CIRCABC links were exposed in the page HTML. "
        "The page may load links dynamically."
    )

print("\nOutput folder:", OUT)
print(
    "\nThis step only discovers official links; "
    "it does not download guessed files."
)
