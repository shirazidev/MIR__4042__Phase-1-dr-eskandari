from requests import Session
from requests.adapters import HTTPAdapter
try:
    # urllib3 v2+
    from urllib3.util.retry import Retry
except Exception:
    # fallback for some environments
    from requests.packages.urllib3.util.retry import Retry

from bs4 import BeautifulSoup
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
from typing import List, Optional, Dict, Any
import json
import time
import random


class IMDbCrawler:
    """
    IMDb Top 250–seeded crawler with resilient networking.
    - Uses a pooled Session with automatic retries for 429/5xx.
    - Global rate limiter across threads (good for slow/fragile networks).
    - Lower default concurrency and sensible timeouts.

    NOTE: Extraction logic mirrors your original. `get_rating` and `get_mpaa`
    are upgraded with multi-strategy fallbacks.
    """

    BASE_URL = "https://www.imdb.com"
    TOP_250_URL = f"{BASE_URL}/chart/top/"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }

    def __init__(
        self,
        crawling_threshold: int = 1000,
        *,
        max_workers: int = 3,
        min_interval: float = 1.0,
        timeout: int = 30,
        retries: int = 6
    ) -> None:
        """
        Parameters
        ----------
        crawling_threshold : int
            Max number of pages to crawl.
        max_workers : int
            Worker threads. Keep small for unreliable connections.
        min_interval : float
            Minimum global delay between *any* two requests (shared by threads).
        timeout : int
            Per-request timeout.
        retries : int
            Total retry attempts for transient HTTP/connection errors.
        """
        self.crawling_threshold = crawling_threshold
        self.not_crawled: deque[str] = deque()
        self.crawled: List[Dict[str, Any]] = []
        self.added_ids: set[str] = set()

        self.add_list_lock = Lock()
        self.add_queue_lock = Lock()

        self.max_workers = max_workers
        self.min_interval = float(min_interval)
        self.timeout = timeout
        self.session = self._build_session(retries)

        self._last_request_ts = 0.0
        self._rate_lock = Lock()

    # --------------------------- Networking -----------------------------

    def _build_session(self, retries: int) -> Session:
        """
        Create a requests.Session with retry/backoff and connection pooling.
        """
        s = Session()

        try:
            retry = Retry(
                total=retries,
                connect=retries,
                read=retries,
                status=retries,
                backoff_factor=1.0,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=frozenset({"GET"}),
                raise_on_status=False,
            )
        except TypeError:
            retry = Retry(
                total=retries,
                connect=retries,
                read=retries,
                status=retries,
                backoff_factor=1.0,
                status_forcelist=[429, 500, 502, 503, 504],
                method_whitelist=frozenset({"GET"}),
                raise_on_status=False,
            )

        adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)
        s.mount("https://", adapter)
        s.mount("http://", adapter)

        s.headers.update({
            "User-Agent": self.headers.get("User-Agent", ""),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": "https://www.imdb.com/",
        })
        return s

    def crawl(self, url: str):
        """
        GET with pooling, retries, timeout, and a GLOBAL rate limit shared by threads.
        Returns Response on 200, else None after retries are exhausted.
        """
        with self._rate_lock:
            now = time.time()
            wait_time = self.min_interval - (now - self._last_request_ts)
            if wait_time > 0:
                time.sleep(wait_time)
            time.sleep(random.uniform(0.05, 0.25))  # jitter
            self._last_request_ts = time.time()

        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                return resp
            print(f"Failed to crawl {url}: Status code {resp.status_code}")
        except Exception as e:
            print(f"An error occurred while crawling {url}: {e}")
        return None

    # ----------------------------- Seeds --------------------------------

    def extract_top_250(self) -> None:
        """
        Seed the queue from the IMDb Top 250 page.
        """
        try:
            response = self.crawl(self.TOP_250_URL)
            if response is None:
                print("Failed to retrieve the Top 250 movies list.")
                return

            soup = BeautifulSoup(response.content, "html.parser")
            movies = soup.find_all("li", class_="ipc-metadata-list-summary-item")

            self.not_crawled.clear()
            self.added_ids.clear()

            for movie in movies:
                link = movie.find("a", href=True)
                if not link:
                    continue
                href = link.get("href", "")
                # Expecting /title/<id>/...
                parts = href.split("/")
                if len(parts) > 2:
                    movie_id = parts[2]
                    if movie_id and movie_id not in self.added_ids:
                        self.not_crawled.append(f"{self.BASE_URL}/title/{movie_id}/")
                        self.added_ids.add(movie_id)
        except Exception as e:
            print(f"An error occurred while extracting the Top 250 movies: {e}")

    # ----------------------------- Public --------------------------------

    def start_crawling(self) -> None:
        """
        Crawl titles starting from Top 250 until threshold or queue exhaustion.
        """
        self.extract_top_250()

        futures = []
        crawled_counter = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while crawled_counter < self.crawling_threshold and self.not_crawled:
                url = self.not_crawled.popleft()
                futures.append(executor.submit(self.crawl_page_info, url, crawled_counter))
                crawled_counter += 1

                if not self.not_crawled:
                    wait(futures)
                    futures.clear()

            if futures:
                wait(futures)

    def crawl_page_info(self, url: str, crawler_counter: int) -> Optional[Dict[str, Any]]:
        """
        Crawl a single title page and extract structured movie information.
        Related links are appended to the queue (same logic as original).
        """
        response = self.crawl(url)
        if response is None:
            print(f"Failed to crawl page: {url}")
            return None

        soup = BeautifulSoup(response.content, "html.parser")

        movie = self.get_imdb_instance()
        movie["id"] = self.get_id_from_URL(url)
        movie["title"] = self.get_title(soup)
        movie["first_page_summary"] = self.get_first_page_summary(soup)
        movie["related_links"] = self.get_related_links(soup)
        movie["mpaa"] = self.get_mpaa(soup)
        movie["budget"] = self.get_budget(soup)
        movie["gross_worldwide"] = self.get_gross_worldwide(soup)
        movie["rating"] = self.get_rating(soup)
        movie["directors"] = self.get_director(soup)
        movie["writers"] = self.get_writers(soup)
        movie["stars"] = self.get_stars(soup)
        movie["release_year"] = self.get_release_year(soup)
        movie["genres"] = self.get_genres(soup)
        movie["languages"] = self.get_languages(soup)
        movie["countries_of_origin"] = self.get_countries_of_origin(soup)
        movie["summaries"] = self.get_summaries(url)
        movie["synopsis"] = self.get_synopsis(url)
        movie["reviews"] = self.get_reviews_with_scores(url)

        with self.add_list_lock:
            self.crawled.append(movie)

        related_links = self.get_related_links(soup)
        if related_links:
            with self.add_queue_lock:
                for link in related_links:
                    movie_id = self.get_id_from_URL(link)
                    if movie_id and movie_id not in self.crawled and movie_id not in self.added_ids:
                        self.not_crawled.append(link)
                        self.added_ids.add(movie_id)

        print(crawler_counter, f"Successfully crawled and processed: {url}")
        return movie

    # --------------------------- Serialization ---------------------------

    def write_to_file_as_json(self) -> None:
        """
        Write crawled results and remaining queue to JSON files.
        """
        try:
            with open("IMDB_crawled.json", "w", encoding="utf-8") as f:
                json.dump(self.crawled, f, ensure_ascii=False, indent=4)
            with open("IMDB_not_crawled.json", "w", encoding="utf-8") as f:
                json.dump(list(self.not_crawled), f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"An error occurred while writing to JSON file: {e}")

    def read_from_file_as_json(self) -> None:
        """
        Read crawled results and queue from JSON files (if present).
        """
        try:
            with open("IMDB_crawled.json", "r", encoding="utf-8") as f:
                self.crawled = json.load(f)
        except FileNotFoundError:
            print("IMDB_crawled.json not found, initializing an empty list.")
            self.crawled = []

        try:
            with open("IMDB_not_crawled.json", "r", encoding="utf-8") as f:
                self.not_crawled = deque(json.load(f))
        except FileNotFoundError:
            print("IMDB_not_crawled.json not found, initializing an empty queue.")
            self.not_crawled = deque()

    # ------------------------- Extraction helpers ------------------------

    @staticmethod
    def get_imdb_instance() -> Dict[str, Any]:
        return {
            "id": None,
            "title": None,
            "first_page_summary": None,
            "release_year": None,
            "mpaa": None,
            "budget": None,
            "gross_worldwide": None,
            "rating": None,
            "directors": None,
            "writers": None,
            "stars": None,
            "related_links": None,
            "genres": None,
            "languages": None,
            "countries_of_origin": None,
            "summaries": None,
            "synopsis": None,
            "reviews": None,
        }

    @staticmethod
    def get_id_from_URL(url: str) -> Optional[str]:
        """
        Extract the IMDB title id from a title URL.
        Example: https://www.imdb.com/title/tt0111161/ -> tt0111161
        """
        try:
            parts = url.split("/")
            title_index = parts.index("title")
            if title_index + 1 < len(parts):
                return parts[title_index + 1]
            print("Error: 'title' segment found, but no ID follows in the URL.")
        except ValueError:
            print("Error: The URL does not contain the 'title' segment.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        return None

    def get_summary_link(self, url: str) -> Optional[str]:
        """
        /title/<id>/ -> /title/<id>/plotsummary
        """
        try:
            movie_id = self.get_id_from_URL(url)
            if movie_id:
                return f"{self.BASE_URL}/title/{movie_id}/plotsummary"
            print("Failed to extract movie ID from URL.")
        except Exception as e:
            print(f"Failed to get summary link: {e}")
        return None

    def get_review_link(self, url: str) -> Optional[str]:
        """
        /title/<id>/ -> /title/<id>/reviews
        """
        try:
            movie_id = self.get_id_from_URL(url)
            if movie_id:
                return f"{self.BASE_URL}/title/{movie_id}/reviews"
            print("Failed to extract movie ID from URL.")
        except Exception as e:
            print(f"Failed to get review link: {e}")
        return None

    @staticmethod
    def get_title(soup: BeautifulSoup) -> Optional[str]:
        try:
            tag = soup.find("h1")
            return tag.text.strip() if tag else None
        except Exception as e:
            print(f"Failed to get title: {e}")
            return None

    @staticmethod
    def get_first_page_summary(soup: BeautifulSoup) -> str:
        try:
            tag = soup.find("span", {"data-testid": "plot-xl"})
            return tag.text.strip() if tag else "Summary not found."
        except Exception as e:
            print(f"Failed to get summary: {e}")
            return "Failed to extract summary."

    @staticmethod
    def _get_principal_names(soup: BeautifulSoup, label_text: str) -> List[str]:
        """
        Extract names (Director, Writers, Stars) from principal credit blocks.
        """
        try:
            names: List[str] = []
            blocks = soup.find_all("li", {"data-testid": "title-pc-principal-credit"})
            for block in blocks:
                for tag_name, cls in [
                    ("span", "ipc-metadata-list-item__label"),
                    ("a", "ipc-metadata-list-item__label"),
                ]:
                    label_tag = block.find(tag_name, class_=cls)
                    if label_tag and label_tag.text.strip() == label_text:
                        for a in block.find_all("a", class_="ipc-metadata-list-item__list-content-item"):
                            txt = a.text.strip()
                            if txt:
                                names.append(txt)
            return list(set(names)) if names else []
        except Exception as e:
            print(f"Failed to get {label_text.lower()}: {e}")
            return []

    def get_director(self, soup: BeautifulSoup) -> List[str]:
        return self._get_principal_names(soup, "Director")

    def get_writers(self, soup: BeautifulSoup) -> List[str]:
        return self._get_principal_names(soup, "Writers")

    def get_stars(self, soup: BeautifulSoup) -> List[str]:
        return self._get_principal_names(soup, "Stars")

    @staticmethod
    def get_related_links(soup: BeautifulSoup) -> List[str]:
        try:
            related_links: List[str] = []
            section = soup.find("section", {"data-testid": "MoreLikeThis"})
            if not section:
                return related_links
            links = section.find_all("a", class_="ipc-lockup-overlay ipc-focusable")
            for link in links:
                href = link.get("href")
                if href:
                    related_links.append(f"https://www.imdb.com{href}")
            return related_links
        except Exception as e:
            print(f"Failed to get related links: {e}")
            return []

    def get_summaries(self, url: str) -> List[str]:
        summaries: List[str] = []
        try:
            summaries_url = self.get_summary_link(url)
            if not summaries_url:
                return summaries
            resp = self.crawl(summaries_url)
            if not resp:
                return summaries
            soup = BeautifulSoup(resp.content, "html.parser")
            section = soup.find("div", {"data-testid": "sub-section-summaries"})
            if not section:
                return summaries
            for li in section.find_all("li"):
                div = li.find("div", class_="ipc-html-content-inner-div")
                if div and div.text:
                    summaries.append(div.text.strip())
        except Exception as e:
            print(f"Failed to get summaries: {e}")
        return summaries

    def get_synopsis(self, url: str) -> List[str]:
        synopsis: List[str] = []
        try:
            synopsis_url = self.get_summary_link(url)
            if not synopsis_url:
                print("Synopsis url not found.")
                return []
            resp = self.crawl(synopsis_url)
            if not resp:
                print("Failed to retrieve the synopsis page.")
                return []
            soup = BeautifulSoup(resp.content, "html.parser")
            blocks = soup.find_all("div", {"data-testid": "sub-section-synopsis"})
            if blocks:
                for div in blocks:
                    if div.text:
                        synopsis.append(div.text.strip())
                return synopsis
            print("Synopsis content not found.")
            return []
        except Exception as e:
            print(f"Failed to get synopsis: {e}")
            return []

    def get_reviews_with_scores(self, url: str) -> List[List[str]]:
        """
        Return a list of [review_text, score] pairs. Score may be "N/A".
        """
        result: List[List[str]] = []
        try:
            reviews_url = self.get_review_link(url)
            if not reviews_url:
                print("review url not found.")
                return []
            resp = self.crawl(reviews_url)
            if not resp:
                print("Failed to retrieve the review page.")
                return []
            soup = BeautifulSoup(resp.content, "html.parser")
            list_div = soup.find("div", class_="lister-list")
            if not list_div:
                return []
            items = list_div.find_all("div", class_="lister-item")
            for item in items:
                text_div = item.find("div", class_="text show-more__control")
                review_text = text_div.text.strip() if text_div and text_div.text else ""
                score_span = item.find("span", class_="rating-other-user-rating")
                score = score_span.find("span").text.strip() if score_span else "N/A"
                result.append([review_text, score])
        except Exception as e:
            print(f"Failed to get reviews: {e}")
        return result

    @staticmethod
    def get_genres(soup: BeautifulSoup) -> List[str]:
        genres: List[str] = []
        try:
            scroller = soup.find("div", class_="ipc-chip-list__scroller")
            if not scroller:
                return genres
            for span in scroller.find_all("span", class_="ipc-chip__text"):
                txt = span.text.strip()
                if txt:
                    genres.append(txt)
        except Exception as e:
            print(f"Failed to get genres: {e}")
        return genres

    @staticmethod
    def get_rating(soup: BeautifulSoup) -> str:
        """
        Robust rating extraction:
        1) data-testid='hero-rating-bar__aggregate-rating__score' (current DOM)
        2) legacy itemprop='ratingValue'
        3) JSON-LD aggregateRating.ratingValue
        """
        try:
            container = soup.find("div", {"data-testid": "hero-rating-bar__aggregate-rating__score"})
            if container:
                text = container.get_text(strip=True)
                if text:
                    return text.split("/")[0]
        except Exception as e:
            print(f"Primary rating selector failed: {e}")

        try:
            span = soup.find(attrs={"itemprop": "ratingValue"})
            if span and span.text.strip():
                return span.text.strip()
        except Exception as e:
            print(f"Legacy rating selector failed: {e}")

        try:
            scripts = soup.find_all("script", type="application/ld+json")
            for sc in scripts:
                if not sc.string:
                    continue
                try:
                    data = json.loads(sc.string.strip())
                except Exception:
                    continue

                def extract_from(obj):
                    if isinstance(obj, dict):
                        agg = obj.get("aggregateRating")
                        if isinstance(agg, dict):
                            val = agg.get("ratingValue") or agg.get("ratingvalue")
                            if val is not None:
                                return str(val)
                    return None

                val = None
                if isinstance(data, list):
                    for item in data:
                        val = extract_from(item)
                        if val:
                            break
                else:
                    val = extract_from(data)

                if val:
                    return val
        except Exception as e:
            print(f"JSON-LD rating parse failed: {e}")

        return "Rating not found."

    @staticmethod
    def get_mpaa(soup: BeautifulSoup) -> str:
        """
        Robust MPAA/Certificate extraction:
        1) Look for a 'Certificate' row in the title details block and read its value.
        2) Fallback to the direct certificates link text (parentalguide/certificates).
        3) Fallback to JSON-LD 'contentRating'.
        """
        try:
            items = soup.find_all("li", class_="ipc-metadata-list__item")
            for li in items:
                label = li.find(["span", "a"], class_="ipc-metadata-list-item__label")
                if label and label.get_text(strip=True) in {"Certificate", "Certification"}:
                    val = li.find("a", class_="ipc-metadata-list-item__list-content-item")
                    if not val:
                        val = li.find("span", class_="ipc-metadata-list-item__list-content-item")
                    if val and val.get_text(strip=True):
                        return val.get_text(strip=True)
        except Exception as e:
            print(f"Certificate row parse failed: {e}")

        try:
            link = soup.find("a", href=lambda h: h and "parentalguide/certificates" in h)
            if link and link.get_text(strip=True):
                return link.get_text(strip=True)
        except Exception as e:
            print(f"Certificates link parse failed: {e}")

        try:
            scripts = soup.find_all("script", type="application/ld+json")
            for sc in scripts:
                if not sc.string:
                    continue
                try:
                    data = json.loads(sc.string.strip())
                except Exception:
                    continue

                def extract_content_rating(obj):
                    if isinstance(obj, dict):
                        cr = obj.get("contentRating") or obj.get("contentrating")
                        if cr:
                            return str(cr)
                    return None

                val = None
                if isinstance(data, list):
                    for item in data:
                        val = extract_content_rating(item)
                        if val:
                            break
                else:
                    val = extract_content_rating(data)

                if val:
                    return val
        except Exception as e:
            print(f"JSON-LD contentRating parse failed: {e}")

        return "MPAA rating not found."

    @staticmethod
    def get_release_year(soup: BeautifulSoup) -> str:
        try:
            tag = soup.find("a", href=lambda h: h and "releaseinfo" in h)
            return tag.text.strip() if tag else "Release year not found."
        except Exception as e:
            print(f"Failed to get release year: {e}")
            return "Failed to extract release year."

    @staticmethod
    def get_languages(soup: BeautifulSoup) -> List[str]:
        try:
            languages: List[str] = []
            section = soup.find("li", {"data-testid": "title-details-languages"})
            if section:
                for a in section.find_all("a", class_="ipc-metadata-list-item__list-content-item"):
                    txt = a.text.strip()
                    if txt:
                        languages.append(txt)
            return languages if languages else ["Languages not found."]
        except Exception as e:
            print(f"Failed to get languages: {e}")
            return ["Failed to extract languages."]

    @staticmethod
    def get_countries_of_origin(soup: BeautifulSoup) -> List[str]:
        try:
            countries: List[str] = []
            for a in soup.find_all("a", href=lambda h: h and "country_of_origin" in h):
                txt = a.text.strip()
                if txt:
                    countries.append(txt)
            return countries if countries else ["No countries of origin found."]
        except Exception as e:
            print(f"Failed to get countries of origin: {e}")
            return ["Failed to extract countries of origin."]

    @staticmethod
    def get_budget(soup: BeautifulSoup) -> str:
        try:
            li = soup.find("li", {"data-testid": "title-boxoffice-budget"})
            if not li:
                return "Budget tag not found."
            span = li.find("span", class_="ipc-metadata-list-item__list-content-item")
            return span.text.strip() if span else "Budget not found."
        except Exception as e:
            print(f"Failed to get budget: {e}")
            return "Failed to extract budget."

    @staticmethod
    def get_gross_worldwide(soup: BeautifulSoup) -> str:
        try:
            li = soup.find("li", {"data-testid": "title-boxoffice-cumulativeworldwidegross"})
            if not li:
                return "Gross worldwide tag not found."
            span = li.find("span", class_="ipc-metadata-list-item__list-content-item")
            return span.text.strip() if span else "Gross worldwide not found."
        except Exception as e:
            print(f"Failed to get gross worldwide: {e}")
            return "Failed to extract gross worldwide."


def main() -> None:
    imdb_crawler = IMDbCrawler(
        crawling_threshold=10,
        max_workers=20,
        min_interval=1.0,
        timeout=30,
        retries=6,
    )
    # imdb_crawler.read_from_file_as_json()
    imdb_crawler.start_crawling()
    imdb_crawler.write_to_file_as_json()


if __name__ == "__main__":
    main()
