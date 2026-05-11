import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_: Any):
        return iterable

try:
    import spacy
except ModuleNotFoundError:
    spacy = None

# Configure production-ready logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("preprocessing.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


_BASIC_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "her", "his", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "this", "to", "was", "were", "with", "who", "when", "where",
}


class _RegexToken:
    """Small spaCy-token compatible fallback used when local deps are unavailable."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.lemma_ = self._lemma(text)
        self.is_stop = text in _BASIC_STOP_WORDS
        self.is_punct = False
        self.is_alpha = text.isalpha()
        self.like_num = text.isdigit()

    @staticmethod
    def _lemma(text: str) -> str:
        if text.endswith("ies") and len(text) > 4:
            return text[:-3] + "y"
        if text.endswith("ing") and len(text) > 5:
            return text[:-3]
        if text.endswith("ed") and len(text) > 4:
            return text[:-2]
        if text.endswith("s") and len(text) > 3:
            return text[:-1]
        return text


class _RegexNLP:
    def __call__(self, text: str) -> List[_RegexToken]:
        return [_RegexToken(match.group(0)) for match in re.finditer(r"[a-z0-9]+", text.lower())]


class DataProcessor:
    """
    Field-preserving text processor for movie data.

    Why this design?
    - Maintains field boundaries for fielded inverted index
    - Processes only required fields (title, summaries, genres, stars)
    - Handles large datasets (1000 movies) with batch processing
    - Preserves original field information for downstream weighting

    Note: No field weighting here - that's for search phase, not preprocessing.

    Assignment note:
    You must implement the TODO parts below. Do not rename methods or change their
    signatures — the autograder imports this class directly.
    """

    # ---------------- DO NOT CHANGE THE SIGNATURE ----------------
    def __init__(
        self,
        input_path: str = "IMDB_crawled.json",
        output_path: str = "fielded_processed_tokens.json",
        nlp_model: str = "en_core_web_sm",
        batch_size: int = 100,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.nlp = self._load_spacy_model(nlp_model)
        self.batch_size = batch_size

        # Only process these fields
        self.indexed_fields = ["title", "summaries", "genres", "stars"]

        # Placeholder strings to filter out (lowercased during cleaning)
        self.placeholder_strings = [
            "summary not found.",
            "failed to extract summary.",
            "languages not found.",
            "failed to extract languages.",
            "release year not found.",
            "failed to extract release year.",
        ]

    # ---------------- PROVIDED HELPER (you may keep as-is) ----------------
    def _load_spacy_model(self, model_name: str):
        """Load spaCy model with retry logic for production resilience."""
        if spacy is None:
            logger.warning("spaCy is not installed. Falling back to regex tokenization.")
            return _RegexNLP()

        try:
            return spacy.load(model_name)
        except OSError:
            logger.warning(f"spaCy model '{model_name}' not found. Falling back to regex tokenization.")
            return _RegexNLP()

    # ---------------- STUDENT TODO ----------------
    def _clean_text(self, text: str) -> str:
        """
        Normalize raw text while preserving meaning for movie-specific terms.

        Requirements:
        - Lowercase the text
        - Remove URLs (e.g., https://… or www.…)
        - Remove any of the placeholder strings listed in self.placeholder_strings
        - Handle common compound movie terms by keeping BOTH forms. Examples:
            "spider-man" -> "spiderman spider man"
            "iron-man" -> "ironman iron man"
            "captain-america" -> "captainamerica captain america"
            "fast-furious" -> "fastfurious fast furious"
        - Replace remaining hyphens with spaces
        - Remove non-alphanumeric characters EXCEPT whitespace and basic punctuation [.,!?;:]
        - Collapse multiple spaces to a single space and strip leading/trailing spaces

        Returns:
            Cleaned string (may be empty).
        """
        # Lowercase the text
        if not text or not isinstance(text, str):
            return ""
        text = text.lower()

        # Remove URLs
        text = re.sub(r'https?://[^\s]+|www\.[^\s]+', '', text)

        # Remove placeholder strings
        for ph in self.placeholder_strings:
            text = text.replace(ph.lower(), "")

        # Handle compound movie terms
        compounds = {
            "spider-man": "spiderman spider man",
            "iron-man": "ironman iron man",
            "captain-america": "captainamerica captain america",
            "fast-furious": "fastfurious fast furious"
        }
        for k, v in compounds.items():
            text = text.replace(k, v)

        # Replace remaining hyphens with spaces
        text = text.replace('-', ' ')

        # Remove non-alphanumeric except whitespace and basic punctuation [.,!?;:]
        text = re.sub(r'[^a-z0-9\s.,!?;:]', '', text)

        # Collapse multiple spaces to single space and strip
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    # ---------------- STUDENT TODO ----------------
    def _tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Convert text into tokens using spaCy, then lemmatize.

        Requirements:
        - Skip stop words and punctuation tokens
        - Keep tokens that are alphabetic OR look like numbers (years matter!)
        - Use token.lemma_ (spaCy lemmatizer)
        - Filter very short lemmas (length <= 2) EXCEPT allow {'tv','dc','mc'}
        - Return a list of lemmas

        Returns:
            List[str]: cleaned lemmas
        """
        # --------- TODO: Implement tokenization + lemmatization ---------
        doc = self.nlp(text)
        lemmas = []
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
            if token.is_alpha or token.like_num:
                lemma = token.lemma_
                if len(lemma) <= 2 and lemma not in {'tv', 'dc', 'mc'}:
                    continue
                lemmas.append(lemma)
        return lemmas

    # ---------------- STUDENT TODO ----------------
    def _process_movie(self, movie: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Process a single movie entry while preserving field boundaries.

        Requirements:
        - Only consider fields in self.indexed_fields
        - For 'summaries' (list of strings), join with spaces before cleaning
        - For other list fields, join with ', ' (comma+space)
        - Skip fields that are missing/empty
        - Skip if a field's text contains any placeholder string (case-insensitive)
        - Use _clean_text -> _tokenize_and_lemmatize pipeline
        - Return a dict[field] = list_of_tokens (omit fields with zero tokens)

        Returns:
            Dict[str, List[str]]
        """
        # --------- TODO: Implement per-field processing ---------
        result = {}
        for field in self.indexed_fields:
            if field not in movie or not movie[field]:
                continue
                
            raw_data = movie[field]
            if isinstance(raw_data, list):
                if field == 'summaries':
                    text = ' '.join(str(item) for item in raw_data if item)
                else:
                    text = ', '.join(str(item) for item in raw_data if item)
            else:
                text = str(raw_data)
                
            text_lower = text.lower()
            if any(ph.lower() in text_lower for ph in self.placeholder_strings):
                continue
                
            cleaned_text = self._clean_text(text)
            if not cleaned_text:
                continue
                
            tokens = self._tokenize_and_lemmatize(cleaned_text)
            if tokens:
                result[field] = tokens
                
        return result

    # ---------------- PROVIDED ORCHESTRATION (calls your TODOs) ----------------
    def process(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Main processing pipeline with memory optimization.
        Reads input JSON, processes in batches, saves partial progress,
        writes final JSON, then generates a validation report.
        """
        logger.info(f"Starting field-preserving preprocessing from {self.input_path}")
        start_time = time.time()

        # Load raw data
        try:
            with open(self.input_path, "r", encoding="utf-8") as f:
                raw_data: List[Dict[str, Any]] = json.load(f)
            logger.info(f"Loaded {len(raw_data)} raw movie entries")
        except Exception as e:
            logger.error(f"Failed to load raw data: {e}")
            raise

        processed_data: Dict[str, Dict[str, List[str]]] = {}
        total_batches = (len(raw_data) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(total_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min((batch_idx + 1) * self.batch_size, len(raw_data))
            batch = raw_data[batch_start:batch_end]

            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({batch_start}-{batch_end})")

            for movie in tqdm(batch, desc=f"Batch {batch_idx + 1}"):
                try:
                    movie_id = movie.get("id")
                    if not movie_id:
                        continue

                    field_tokens = self._process_movie(movie)  # <- your implementation
                    if field_tokens:
                        processed_data[movie_id] = field_tokens
                except Exception as e:
                    logger.error(f"Error processing movie {movie.get('id', 'UNKNOWN')}: {e}")

            # Save intermediate progress
            self._save_progress(processed_data)

        # Final save + report
        self._save_final(processed_data)
        self._generate_validation_report(raw_data, processed_data)

        elapsed = time.time() - start_time
        logger.info(f"Preprocessing completed in {elapsed:.2f} seconds")
        return processed_data

    # ---------------- PROVIDED HELPERS (keep as-is) ----------------
    def _save_progress(self, processed_data: Dict[str, Dict[str, List[str]]]) -> None:
        """Save intermediate progress to prevent data loss on failure."""
        try:
            with open(str(self.output_path) + ".tmp", "w", encoding="utf-8") as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save intermediate progress: {e}")

    def _save_final(self, processed_data: Dict[str, Dict[str, List[str]]]) -> None:
        """Save final results with cleanup of temporary files."""
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Processed data saved to {self.output_path}")

            tmp_path = Path(str(self.output_path) + ".tmp")
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception as e:
            logger.error(f"Failed to save final results: {e}")
            raise

    # ---------------- DO NOT CHANGE THIS METHOD ----------------
    def _generate_validation_report(self, raw_data: List[Dict], processed_data: Dict[str, Dict[str, List[str]]]) -> None:
        """
        Generate comprehensive validation report for QA.

        Report includes:
        - Processing statistics
        - Field coverage analysis
        - Token distribution
        - Sample entries for manual verification
        """
        report = [
            "===== FIELD-PRESERVING PREPROCESSING VALIDATION REPORT =====",
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total raw entries: {len(raw_data)}",
            f"Successfully processed: {len(processed_data)}",
            f"Processing success rate: {len(processed_data)/len(raw_data)*100:.2f}%",
            "\n===== FIELD COVERAGE ====="
        ]

        # Field coverage statistics
        field_counts = {field: 0 for field in self.indexed_fields}
        total_tokens = 0
        unique_tokens = set()

        for movie_fields in processed_data.values():
            for field, tokens in movie_fields.items():
                if field in field_counts:
                    field_counts[field] += 1
                    total_tokens += len(tokens)
                    unique_tokens.update(tokens)

        for field, count in field_counts.items():
            report.append(f"  - {field}: {count} movies ({count/len(raw_data)*100:.2f}%)")

        report.extend([
            f"\n===== TOKEN STATISTICS =====",
            f"Total tokens: {total_tokens}",
            f"Unique tokens: {len(unique_tokens)}",
            f"Average tokens per movie: {total_tokens/len(processed_data):.2f}"
        ])

        # Sample validation
        report.append("\n===== SAMPLE VALIDATION =====")

        # Find Shawshank Redemption for sample inspection
        shawshank_id = next((mid for mid, data in processed_data.items()
                            if "shawshank" in mid.lower() or "redemption" in mid.lower()), None)

        if shawshank_id and shawshank_id in processed_data:
            movie_data = processed_data[shawshank_id]
            report.append(f"Sample movie: {shawshank_id}")

            for field, tokens in movie_data.items():
                report.append(f"  - {field} ({len(tokens)} tokens): {tokens[:5]}{'...' if len(tokens) > 5 else ''}")
        else:
            report.append("  - Could not find Shawshank Redemption for sample inspection")

        # Token frequency analysis
        token_freq = {}
        for movie_fields in processed_data.values():
            for tokens in movie_fields.values():
                for token in tokens:
                    token_freq[token] = token_freq.get(token, 0) + 1

        most_common = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        report.append("\n===== MOST COMMON TOKENS =====")
        for token, freq in most_common:
            report.append(f"  - {token}: {freq}")

        report.append("\n===== END OF REPORT =====")

        # Save report
        report_path = self.output_path.parent / "preprocessing_validation.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))

        logger.info(f"Validation report saved to {report_path}")

# ---------------- Entry point (leave as-is) ----------------
def main() -> None:
    """
    Execute preprocessing.
    Usage:
        python preprocess.py
    """
    try:
        processor = DataProcessor(
            input_path="IMDB_crawled.json",
            output_path="fielded_processed_tokens.json",
            batch_size=100,
        )
        processed = processor.process()
        print("\n✅ Field-preserving preprocessing finished.")
        print(f"Total processed movies: {len(processed)}")
        print(f"Output saved to: {processor.output_path}")
        print(f"Validation report: {processor.output_path.parent / 'preprocessing_validation.txt'}")
    except NotImplementedError as nie:
        logger.error(f"Unimplemented part of the assignment: {nie}")
        raise SystemExit(2)
    except Exception as e:
        logger.exception("Fatal error in preprocessing pipeline")
        print(f"\n❌ Preprocessing failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
