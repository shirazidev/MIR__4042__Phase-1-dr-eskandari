# --- Query Parser: Field Weight Extraction w/ Spelling Correction (Student Version) ---

import json
import logging
from typing import Dict, Tuple, Any, Optional, List
import re
import time

# Configure consistent logging with project standards
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("query_parser.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QueryParser:
    """
    Parses user queries to extract field weights and search terms.

    Assignment note:
    Implement ONLY the TODO parts. Do NOT change method names/signatures.
    """

    def __init__(self,
                 allowed_fields: Optional[List[str]] = None,
                 default_weight: float = 1.0,
                 inverted_index_path: str = "inverted_index.json") -> None:
        # Default fields (title, summaries, genres, stars)
        self.allowed_fields = allowed_fields or ["title", "summaries", "genres", "stars"]
        self.default_weight = default_weight

        # Field aliases
        self.field_aliases = {"summary": "summaries"}

        # Default weights (copied on each parse)
        self.default_weights = {field: self.default_weight for field in self.allowed_fields}

        # Regex for field:weight
        self.field_weight_pattern = re.compile(r'^([a-zA-Z_]+):(-?[0-9]*\.?[0-9]+)$')

        # Vocab + stats for spelling correction
        self.vocabulary: set = set()
        self.term_df: Dict[str, int] = {}
        self.total_docs: int = 1
        self._load_vocabulary(inverted_index_path)

        logger.debug(f"QueryParser initialized with fields: {', '.join(self.allowed_fields)}")
        logger.debug(f"Vocabulary size: {len(self.vocabulary)} terms; total_docs={self.total_docs}")

    # ---------------- PROVIDED (keep) ----------------
    def _load_vocabulary(self, inverted_index_path: str) -> None:
        """Populate self.vocabulary (set), self.term_df (token->df), self.total_docs (int)."""
        try:
            with open(inverted_index_path, 'r', encoding='utf-8') as f:
                idx = json.load(f)
            index = idx.get("index", {})
            self.vocabulary = set(index.keys())
            self.term_df = {
                tok: (data.get("df", 0) if isinstance(data, dict) else 0)
                for tok, data in index.items()
            }
            meta = idx.get("metadata", {})
            self.total_docs = int(meta.get("total_documents", 1)) or 1
            logger.info(f"Vocabulary loaded: {len(self.vocabulary)} terms; total_docs={self.total_docs}")
        except Exception as e:
            logger.error(f"Failed to load vocabulary: {str(e)}")
            logger.warning("Spelling correction will degrade (no DF stats)")
            self.vocabulary = set()
            self.term_df = {}
            self.total_docs = 1

    # ---------------- STUDENT TODO ----------------
    def _morph_variants(self, term: str) -> List[str]:
        """
        Produce simple plural/singular variants of `term` and return only those present in `self.vocabulary`.

        Examples (not exhaustive):
        - dogs -> dog
        - boxes -> box
        - stories -> story
        - baby -> babies
        - bus -> buses

        Return:
            List[str]: variants found in vocabulary (may be empty).
        """
        # --------- TODO: implement morphological variant generation ---------
        term = term.lower().strip()
        candidates = set()

        base = self._plural_base(term)
        if base:
            candidates.add(base)

        if term.endswith("y") and len(term) > 1 and term[-2] not in "aeiou":
            candidates.add(term[:-1] + "ies")
        if term.endswith(("s", "x", "z", "ch", "sh")):
            candidates.add(term + "es")
        candidates.add(term + "s")

        return [word for word in candidates if word in self.vocabulary]

    # ---------------- STUDENT TODO ----------------
    def _plural_class(self, w: str) -> str:
        """
        Classify plural 'shape' of a word for tie-breaking:
        - return one of {"IES","ES","S","NONE"} based on suffix patterns.
        """
        # --------- TODO: implement plural class detection ---------
        w = w.lower()
        if w.endswith("ies") and len(w) > 3:
            return "IES"
        if w.endswith(("ses", "xes", "zes", "ches", "shes")) and len(w) > 3:
            return "ES"
        if w.endswith("s") and len(w) > 1:
            return "S"
        return "NONE"

    # ---------------- STUDENT TODO ----------------
    def _plural_base(self, w: str) -> str:
        """
        Roughly map plural forms back to a singular base:
        - stories -> story
        - boxes  -> box
        - dogs   -> dog
        Return "" if no plausible base.
        """
        # --------- TODO: implement plural base extraction ---------
        w = w.lower().strip()
        if len(w) <= 2:
            return ""
        if w.endswith("ies") and len(w) > 3:
            return w[:-3] + "y"
        if w.endswith(("ses", "xes", "zes", "ches", "shes")):
            return w[:-2]
        if w.endswith("s") and not w.endswith("ss"):
            return w[:-1]
        return ""

    # ---------------- STUDENT TODO ----------------
    def _is_plausible_plural_of_vocab(self, w: str) -> bool:
        """
        Return True if `w` looks like a plural form AND its singular base is in vocabulary.
        This helps avoid changing user input like 'rings' when 'ring' exists.
        """
        # --------- TODO: implement plural plausibility check using _plural_base and self.vocabulary ---------
        base = self._plural_base(w)
        return bool(base and base in self.vocabulary)

    # ---------------- PROVIDED (keep) ----------------
    def _reasonable_df(self, df: int) -> bool:
        """
        Gate very rare tokens: require DF >= floor(max(2, 0.002 * total_docs)).
        This blocks weird names from stealing corrections.
        """
         # --------- TODO: implement reasonable dfs ---------
        minimum_df = int(max(2, 0.002 * self.total_docs))
        return df >= minimum_df

    # ---------------- STUDENT TODO ----------------
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Compute Levenshtein distance using a memory-light DP (row-based).

        Requirements:
        - Return non-negative integer distance.
        - Use O(min(n,m)) extra memory (no full matrix).
        """
        # --------- TODO: implement Levenshtein DP ---------
        if s1 == s2:
            return 0
        if len(s1) < len(s2):
            s1, s2 = s2, s1

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1, start=1):
            current_row = [i]
            for j, c2 in enumerate(s2, start=1):
                insert_cost = current_row[j - 1] + 1
                delete_cost = previous_row[j] + 1
                replace_cost = previous_row[j - 1] + (0 if c1 == c2 else 1)
                current_row.append(min(insert_cost, delete_cost, replace_cost))
            previous_row = current_row

        return previous_row[-1]

    # ---------------- STUDENT TODO ----------------
    def _correct_spelling(self, term: str) -> str:
        """
        Conservative correction using vocabulary + document frequency:

        Rules:
        - If term is very common/short (e.g., stop-like) or numeric → return as-is.
        - If term ∈ vocabulary → return as-is.
        - If term is a plausible plural of a vocab lemma → return as-is.
        - Else scan vocabulary candidates:
            * Length difference ≤ 2, same first letter (fast prefilter).
            * Levenshtein distance ≤ (1 if len(term) ≥ 5 else 2).
            * Candidate DF must pass _reasonable_df(df).
            * Prefer (distance, same_plural_class?, same_length?, -df) lexicographically.
        - If no suitable candidate → return original term.
        """
        # --------- TODO: implement conservative correction strategy ---------
        term = term.lower().strip()
        lookup_term = term.replace("-", "") if "-" in term else term
        common_words = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for",
            "from", "in", "is", "it", "of", "on", "or", "the", "to", "with"
        }

        if not term or term in common_words or lookup_term.isnumeric() or len(lookup_term) <= 2:
            return term
        if term in self.vocabulary:
            return term
        if lookup_term in self.vocabulary:
            return lookup_term
        if self._is_plausible_plural_of_vocab(lookup_term):
            return term

        variants = self._morph_variants(lookup_term)
        if variants:
            return max(variants, key=lambda word: self.term_df.get(word, 0))

        max_distance = 1 if len(lookup_term) >= 5 else 2
        best_word = term
        best_score = None
        term_class = self._plural_class(lookup_term)

        for candidate in self.vocabulary:
            if not candidate or candidate[0] != lookup_term[0]:
                continue
            if abs(len(candidate) - len(lookup_term)) > 2:
                continue
            df = self.term_df.get(candidate, 0)
            if not self._reasonable_df(df):
                continue

            distance = self._levenshtein_distance(lookup_term, candidate)
            if distance > max_distance:
                continue

            same_plural_class = 0 if self._plural_class(candidate) == term_class else 1
            same_length = 0 if len(candidate) == len(lookup_term) else 1
            score = (distance, same_plural_class, same_length, -df, candidate)
            if best_score is None or score < best_score:
                best_score = score
                best_word = candidate

        return best_word

    # ---------------- STUDENT TODO ----------------
    def parse(self, query: str) -> Tuple[Dict[str, float], str]:
        """
        Parse a user query to extract field weights and search terms.

        Steps:
        1) Start with a copy of self.default_weights.
        2) Preprocess quotes via _preprocess_quotes.
        3) Split into parts by whitespace.
        4) For each part:
           - If matches field:weight:
               • Map aliases (summary→summaries).
               • If field allowed: parse float; clamp negatives to 0.0; set.
               • Else: treat the whole part as a normal term.
           - Else:
               • Correct spelling via _correct_spelling and append to terms.
        5) Return (weights_dict, "terms string" preserving order).
        """
        # --------- TODO: implement query parsing workflow ---------
        weights = self.default_weights.copy()
        terms = []

        query = self._preprocess_quotes(query or "")
        for part in query.split():
            match = self.field_weight_pattern.match(part)
            if match:
                field = self.field_aliases.get(match.group(1), match.group(1))
                if field in self.allowed_fields:
                    weights[field] = max(0.0, float(match.group(2)))
                else:
                    terms.append(self._correct_spelling(part.lower()))
            else:
                terms.append(self._correct_spelling(part.lower()))

        return weights, " ".join(terms)

    # ---------------- STUDENT TODO ----------------
    def _preprocess_quotes(self, query: str) -> str:
        """
        Remove single/double quotes but keep inner content to preserve term order.
        (Advanced handling of phrases is not required here.)
        """
        # --------- TODO: implement basic quote cleanup ---------
        return query.replace('"', " ").replace("'", " ")

    # ---------------- STUDENT TODO ----------------
    def validate_weights(self, weights: Dict[str, float]) -> bool:
        """
        Validate field weights for correctness.

        Checks:
        - All keys belong to self.allowed_fields.
        - All values are numeric (int/float).
        - Warn (or normalize later) if any value is negative.

        Return:
            bool indicating whether weights are acceptable.
        """
        # --------- TODO: implement weight validation ---------
        for field, value in weights.items():
            if field not in self.allowed_fields:
                return False
            if not isinstance(value, (int, float)):
                return False
            if value < 0:
                logger.warning(f"Negative weight for '{field}' will be normalized")
        return True

    # ---------------- PROVIDED (keep) ----------------
    def normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Clamp negative weights to zero (simple normalization)."""
        normalized = weights.copy()
        for field in normalized:
            if normalized[field] < 0:
                logger.info(f"Normalizing negative weight for '{field}' to 0.0")
                normalized[field] = 0.0
        return normalized

    # ---------------- PROVIDED (keep) ----------------
    def parse_and_validate(self, query: str) -> Tuple[Dict[str, float], str]:
        """Parse a query and return normalized, validated weights along with terms."""
        weights, terms = self.parse(query)
        if not self.validate_weights(weights):
            logger.error("Query parsing produced invalid weights - using defaults")
            weights = self.default_weights.copy()
        return self.normalize_weights(weights), terms


def main() -> None:
    """
    Minimal driver for local testing.

    Usage:
        python query_parser.py "stars:1 title:0 summary:0.4 genres:0 drama prison"
    """
    try:
        parser = QueryParser()
        sample_query = "stars:1 title:0 summary:0.4 genres:0 spiedr man"
        weights, terms = parser.parse(sample_query)
        print("\nParsed query:", sample_query)
        print("Field weights:", weights)
        print("Search terms:", terms)
    except NotImplementedError as nie:
        logger.error(f"Unimplemented part of the assignment: {nie}")
        raise SystemExit(2)
    except Exception as e:
        logger.exception("Fatal error in query parser")
        print(f"\nQuery parser failed: {str(e)}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
