import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------
# Utilities & Type Checking
# ------------------------------

def type_check(obj: Any, expected_type: Any) -> bool:
    """Return True if obj matches the expected (possibly typing.List) type.

    Supports concrete types and list[...] recursively. Mirrors original behavior
    but kept here for schema checks.
    """
    if not hasattr(expected_type, "__origin__"):
        return isinstance(obj, expected_type)

    assert expected_type.__origin__ == list, "only list type is supported"
    inner_type = expected_type.__args__[0]
    return isinstance(obj, list) and all(type_check(item, inner_type) for item in obj)


ID_RE = re.compile(r"^tt\d{7,8}$")
TITLE_URL_ID_RE = re.compile(r"/title/(tt\d{7,8})/")
MONEY_RE = re.compile(r"^[^0-9$]*\$?\s?\d{1,3}(,\d{3})*(\.\d+)?(\s*[A-Z]{3})?$")  # loose
RATING_RE = re.compile(r"^(?:10(?:\.0)?|\d(?:\.\d)?)$")  # '8.7', '10', '7'
YEAR_RE = re.compile(r"^\d{4}$")

PLACEHOLDER_STRINGS = {
    "rating": {"Rating not found.", "Failed to extract rating."},
    "mpaa": {"MPAA rating not found.", "Failed to extract MPAA rating."},
    "release_year": {"Release year not found.", "Failed to extract release year."},
    "budget": {"Budget tag not found.", "Budget not found.", "Failed to extract budget."},
    "gross_worldwide": {
        "Gross worldwide tag not found.",
        "Gross worldwide not found.",
        "Failed to extract gross worldwide.",
    },
    "languages": {"Languages not found.", "Failed to extract languages."},
    "countries_of_origin": {
        "No countries of origin found.",
        "Failed to extract countries of origin.",
    },
    "first_page_summary": {"Summary not found.", "Failed to extract summary."},
}


# ------------------------------
# Reporter
# ------------------------------

@dataclass
class TestReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    field_missing_counts: Counter = field(default_factory=Counter)
    field_placeholder_counts: Counter = field(default_factory=Counter)

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def summarize(self, total_movies: int) -> None:
        print("\n=== TEST SUMMARY ===")
        print(f"Total movies: {total_movies}")
        print(f"Errors: {len(self.errors)} | Warnings: {len(self.warnings)}")

        if self.field_missing_counts:
            print("\nMissing/None fields counts:")
            for k, v in self.field_missing_counts.most_common():
                print(f"  - {k}: {v}")

        if self.field_placeholder_counts:
            print("\nPlaceholder-value counts:")
            for k, v in self.field_placeholder_counts.most_common():
                print(f"  - {k}: {v}")

        if self.errors:
            print("\nFirst 10 errors:")
            for e in self.errors[:10]:
                print("  *", e)

        if self.warnings:
            print("\nFirst 10 warnings:")
            for w in self.warnings[:10]:
                print("  -", w)


# ------------------------------
# Validators (per-field & cross-record)
# ------------------------------

EXPECTED_FIELDS: Dict[str, Any] = {
    "id": str,
    "title": str,
    "first_page_summary": str,
    "release_year": str,
    "mpaa": str,
    "budget": str,
    "gross_worldwide": str,
    "rating": str,
    "directors": List[str],
    "writers": List[str],
    "stars": List[str],
    "related_links": List[str],
    "genres": List[str],
    "languages": List[str],
    "countries_of_origin": List[str],
    "summaries": List[str],
    "synopsis": List[str],
    "reviews": List[List[str]],
}


def is_placeholder(field: str, value: Any) -> bool:
    ph = PLACEHOLDER_STRINGS.get(field)
    if ph is None:
        return False
    if isinstance(value, str):
        return (value in ph) or value.startswith("Failed to extract")
    if isinstance(value, list) and value and isinstance(value[0], str):
        return any(v in ph or (isinstance(v, str) and v.startswith("Failed to extract")) for v in value)
    return False


def get_id_from_url(url: str) -> Optional[str]:
    m = TITLE_URL_ID_RE.search(url)
    return m.group(1) if m else None



def validate_schema(movie: Dict[str, Any], idx: int, report: TestReport) -> None:
    for field, expected_type in EXPECTED_FIELDS.items():
        if field not in movie or movie[field] is None:
            report.field_missing_counts[field] += 1
            mid = movie.get("id", f"<index:{idx}>")
            report.warn(f"Missing field '{field}' in movie {mid}")
        else:
            if not type_check(movie[field], expected_type):
                mid = movie.get("id", f"<index:{idx}>")
                report.error(
                    f"Type mismatch for field '{field}' in movie {mid}: "
                    f"expected {expected_type}, got {type(movie[field])}"
                )


def validate_id(movie: Dict[str, Any], idx: int, report: TestReport) -> None:
    mid = movie.get("id")
    if not isinstance(mid, str) or not ID_RE.match(mid or ""):
        report.error(f"Invalid IMDb id at index {idx}: {mid}")


def validate_title(movie: Dict[str, Any], idx: int, report: TestReport) -> None:
    title = movie.get("title")
    if isinstance(title, str):
        if not title.strip():
            report.warn(f"Empty title for movie {movie.get('id', idx)}")
    else:
        report.error(f"Title is not a string for movie {movie.get('id', idx)}: {type(title)}")


def validate_release_year(movie: Dict[str, Any], idx: int, report: TestReport) -> None:
    year = movie.get("release_year")
    if isinstance(year, str):
        if is_placeholder("release_year", year):
            report.field_placeholder_counts["release_year"] += 1
        elif not YEAR_RE.match(year):
            report.warn(f"Suspicious release_year for {movie.get('id', idx)}: '{year}'")
    else:
        report.error(f"release_year is not str for {movie.get('id', idx)}: {type(year)}")


def validate_rating(movie: Dict[str, Any], idx: int, report: TestReport) -> None:
    rating = movie.get("rating")
    if isinstance(rating, str):
        if is_placeholder("rating", rating):
            report.field_placeholder_counts["rating"] += 1
        else:
            if not RATING_RE.match(rating):
                report.warn(f"Unexpected rating format for {movie.get('id', idx)}: '{rating}'")
    else:
        report.error(f"rating is not str for {movie.get('id', idx)}: {type(rating)}")


def validate_money(field: str, movie: Dict[str, Any], idx: int, report: TestReport) -> None:
    value = movie.get(field)
    if isinstance(value, str):
        if is_placeholder(field, value):
            report.field_placeholder_counts[field] += 1
        else:
            if not re.search(r"\d", value):
                report.warn(f"{field} contains no digits for {movie.get('id', idx)}: '{value}'")
    else:
        report.error(f"{field} is not str for {movie.get('id', idx)}: {type(value)}")


def validate_string_list(field: str, movie: Dict[str, Any], idx: int, report: TestReport, min_len: int = 0) -> None:
    value = movie.get(field)
    if not isinstance(value, list):
        report.error(f"{field} is not list for {movie.get('id', idx)}: {type(value)}")
        return

    if not value and min_len > 0:
        report.warn(f"{field} is empty for {movie.get('id', idx)}")

    if is_placeholder(field, value):
        report.field_placeholder_counts[field] += 1

    for i, item in enumerate(value):
        if not isinstance(item, str):
            report.error(f"{field}[{i}] not str for {movie.get('id', idx)}: {type(item)}")
        elif not item.strip():
            report.warn(f"{field}[{i}] empty string for {movie.get('id', idx)}")

    if len(value) != len(set(value)):
        report.warn(f"{field} contains duplicates for {movie.get('id', idx)}")


def validate_related_links(movie: Dict[str, Any], idx: int, report: TestReport) -> None:
    links = movie.get("related_links", [])
    if not isinstance(links, list):
        report.error(f"related_links is not a list for {movie.get('id', idx)}")
        return

    seen = set()
    for j, link in enumerate(links):
        if not isinstance(link, str) or not link.startswith("https://www.imdb.com/"):
            report.warn(f"related_links[{j}] bad URL for {movie.get('id', idx)}: {link}")
            continue
        mid = get_id_from_url(link)
        if not mid or not ID_RE.match(mid):
            report.warn(f"related_links[{j}] missing/invalid id for {movie.get('id', idx)}: {link}")
        if link in seen:
            report.warn(f"related_links[{j}] duplicate link for {movie.get('id', idx)}: {link}")
        seen.add(link)


def validate_reviews(movie: Dict[str, Any], idx: int, report: TestReport) -> None:
    reviews = movie.get("reviews")
    if not isinstance(reviews, list):
        report.error(f"reviews not list for {movie.get('id', idx)}: {type(reviews)}")
        return

    for j, rev in enumerate(reviews):
        if not isinstance(rev, list):
            report.error(f"reviews[{j}] not list for {movie.get('id', idx)}: {type(rev)}")
            continue
        if len(rev) < 2:
            report.warn(f"reviews[{j}] has length < 2 for {movie.get('id', idx)}")
        text = rev[0] if rev else ""
        score = rev[1] if len(rev) > 1 else "N/A"
        if not isinstance(text, str) or not text.strip():
            report.warn(f"reviews[{j}] text empty/non-str for {movie.get('id', idx)}")
        if not isinstance(score, str):
            report.warn(f"reviews[{j}] score non-str for {movie.get('id', idx)}: {type(score)}")
        else:
            if score != "N/A" and not re.match(r"^\d{1,2}$", score):
                report.warn(f"reviews[{j}] score unusual for {movie.get('id', idx)}: '{score}'")
            else:
                if score.isdigit():
                    s = int(score)
                    if s < 0 or s > 10:
                        report.warn(f"reviews[{j}] score out of range 0-10 for {movie.get('id', idx)}: {s}")


def validate_unique_ids(data: List[Dict[str, Any]], report: TestReport) -> None:
    ids = [d.get("id") for d in data if isinstance(d.get("id"), str)]
    counts = Counter(ids)
    dups = [mid for mid, c in counts.items() if c > 1]
    if dups:
        report.error(f"Duplicate IDs found: {dups[:10]}{' ...' if len(dups) > 10 else ''}")


# ------------------------------
# Orchestrator
# ------------------------------

def run_tests(json_file_path: str, min_count: int = 5, fail_on_warning: bool = False) -> int:
    report = TestReport()

    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        report.error(f"Top-level JSON must be a list, got {type(data)}")
        report.summarize(0)
        print("\nFAILED")
        return 1

    if len(data) < min_count:
        report.error(f"Expected at least {min_count} movies, but got {len(data)}")

    for idx, movie in enumerate(data):
        if not isinstance(movie, dict):
            report.error(f"Item at index {idx} is not an object: {type(movie)}")
            continue
        validate_schema(movie, idx, report)
        validate_id(movie, idx, report)
        validate_title(movie, idx, report)
        validate_release_year(movie, idx, report)
        validate_rating(movie, idx, report)
        validate_money("budget", movie, idx, report)
        validate_money("gross_worldwide", movie, idx, report)
        for field in [
            "directors",
            "writers",
            "stars",
            "genres",
            "languages",
            "countries_of_origin",
            "summaries",
            "synopsis",
        ]:
            validate_string_list(field, movie, idx, report)
        validate_related_links(movie, idx, report)
        validate_reviews(movie, idx, report)

    validate_unique_ids(data, report)

    report.summarize(len(data))

    if report.errors:
        print("\nFAILED")
        return 1
    if fail_on_warning and report.warnings:
        print("\nFAILED (warnings treated as failures)")
        return 2
    print("\nPASSED")
    return 0


# ------------------------------
# CLI
# ------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Robust tester for IMDb crawler output JSON.")
    parser.add_argument("json_file", nargs="?", default="IMDB_crawled.json", help="Path to crawler output JSON")
    parser.add_argument("--min-count", type=int, default=10, help="Minimum number of movies expected (default: 800)")
    parser.add_argument(
        "--fail-on-warning", action="store_true", help="Treat warnings as failures (exit code 2)"
    )
    args = parser.parse_args()

    code = run_tests(args.json_file, min_count=args.min_count, fail_on_warning=args.fail_on_warning)
    raise SystemExit(code)


if __name__ == "__main__":
    main()

import sys

sys.argv = ['validator.py', 'IMDB_crawled.json', '--min-count', '50']

if __name__ == "__main__":
    main()
