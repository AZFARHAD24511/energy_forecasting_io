
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

URLS = [
    "https://ec.europa.eu/eurostat/web/esa-supply-use-input-tables/database?node_code=naio_10_fg",
    "https://circabc.europa.eu/ui/group/cec66924-a924-4f91-a0ef-600a0531e3ba/library/460a20fc-5956-44f9-8633-2821eb48dd62",
]

for url in URLS:
    print("\nChecking:", url)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urlopen(req, timeout=30) as response:
            print("Status:", response.status)
            print("Final URL:", response.geturl())
            print("Content type:", response.headers.get("Content-Type"))
            print("First 300 bytes:", response.read(300))

    except (HTTPError, URLError, TimeoutError) as exc:
        print("Request failed:", exc)
