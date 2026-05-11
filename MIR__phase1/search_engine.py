import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
import math
from collections import defaultdict
import time

# Import the parser from your project
from query_parser import QueryParser

# Configure production-grade logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("search_engine.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SearchEngine:
    """
    Field-weighted search engine using BM25 ranking with phrase support.

    Assignment note:
    Implement the TODO parts below. Do NOT change method names/signatures
    or the expected data formats; the autograder relies on them.
    """

    def __init__(self,
                 inverted_index_path: str = "inverted_index.json",
                 k1: float = 1.2,
                 b: float = 0.75,
                 phrase_boost: float = 1.5) -> None:
        self.inverted_index_path = Path(inverted_index_path)
        self.k1 = k1
        self.b = b
        self.phrase_boost = phrase_boost

        # Load inverted index
        self._load_inverted_index()

        # Initialize query parser
        self.query_parser = QueryParser()

        # Precompute document length statistics
        self._compute_doc_length_stats()

        logger.info(f"SearchEngine initialized with k1={k1}, b={b}, phrase_boost={phrase_boost}")
        logger.info(f"Index contains {len(self.inverted_index['index'])} unique tokens across {self.total_docs} documents")

    # ---------------- PROVIDED (keep) ----------------
    def _load_inverted_index(self) -> None:
        """Load inverted index with comprehensive error handling."""
        start_time = time.time()
        try:
            with open(self.inverted_index_path, 'r', encoding='utf-8') as f:
                self.inverted_index = json.load(f)
            self.total_docs = self.inverted_index["metadata"]["total_documents"]
            self.indexed_fields = self.inverted_index["metadata"]["indexed_fields"]
            logger.info(f"Loaded inverted index from {self.inverted_index_path}")
        except FileNotFoundError:
            logger.error(f"Inverted index file not found at {self.inverted_index_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in inverted index: {str(e)}")
            raise
        except KeyError as e:
            logger.error(f"Missing required field in inverted index: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading inverted index: {str(e)}")
            raise
        logger.debug(f"Inverted index loaded in {time.time() - start_time:.6f} seconds")

    # ---------------- STUDENT TODO ----------------
    def _compute_doc_length_stats(self) -> None:
        """
        Compute document length statistics for BM25 normalization.

        Requirements:
        - For each posting in each token: accumulate TF per doc_id.
        - Save per-doc lengths in self.doc_lengths (dict[str,int]).
        - Compute self.avg_doc_length = average of all doc lengths; use 1.0 if empty.
        """
        # --------- TODO: Implement per-doc length accumulation and average ---------
        self.doc_lengths = defaultdict(int)
        for token, token_data in self.inverted_index["index"].items():
            for posting in token_data["postings"]:
                self.doc_lengths[posting["doc_id"]] += posting["tf"]
                
        total_len = sum(self.doc_lengths.values())
        num_docs = len(self.doc_lengths)
        self.avg_doc_length = total_len / num_docs if num_docs > 0 else 1.0

    # ---------------- STUDENT TODO ----------------
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Execute search with field-weighted BM25 + phrase boosts.

        Implement the pipeline:
        1) Parse query → (field_weights, terms). If terms empty → return [].
        2) Tokenize terms to lowercased whitespace-split list.
        3) For each token present in index:
           - Compute IDF = ln((N - df + 0.5)/(df + 0.5) + 1).
           - For each posting:
               * Get tf, field, doc_id, doc_len.
               * weight = field_weights.get(field, 1.0); if weight <= 0 → use small value (e.g., 0.01).
               * BM25 core = (tf*(k1+1)) / (tf + k1*(1 - b + b*(doc_len/avg_doc_length))).
               * token_score = weight * idf * core.
               * Accumulate into doc_scores[doc_id] and detailed_scores[doc_id][token].
        4) Identify phrases and apply boosts to doc_scores (see helpers below).
        5) Sort by score desc, return top-k as (doc_id, score, detailed_token_scores).
        """
        start_time = time.time()

        # --------- TODO: Implement BM25 scoring + phrase boosts + top-k selection ---------
        field_weights, search_terms = self.query_parser.parse(query)
        if not search_terms.strip():
            return []
            
        query_tokens = search_terms.lower().split()
        phrase_terms, phrase_positions = self._identify_phrases(query_tokens)
        
        doc_scores = defaultdict(float)
        detailed_scores = defaultdict(lambda: defaultdict(float))
        
        N = self.total_docs
        avg_dl = self.avg_doc_length
        k1 = self.k1
        b = self.b
        
        for token in query_tokens:
            if token not in self.inverted_index["index"]:
                continue
                
            entry = self.inverted_index["index"][token]
            df = entry["df"]
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            
            for posting in entry["postings"]:
                doc_id = posting["doc_id"]
                field = posting["field"]
                tf = posting["tf"]
                doc_len = self.doc_lengths.get(doc_id, avg_dl)
                
                weight = field_weights.get(field, 1.0)
                if weight <= 0:
                    weight = 0.01
                    
                core = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (doc_len / avg_dl)))
                token_score = weight * idf * core
                
                doc_scores[doc_id] += token_score
                detailed_scores[doc_id][token] += token_score
                
        if phrase_terms:
            doc_scores = self._apply_phrase_boosts(doc_scores, phrase_terms, phrase_positions, detailed_scores)
            
        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(doc_id, score, dict(detailed_scores[doc_id])) for doc_id, score in sorted_results]

    # ---------------- STUDENT TODO ----------------
    def _identify_phrases(self, tokens: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, List[int]]]:
        """
        Identify likely phrases (2–3 words) from the token list.

        Guidelines:
        - Start with a few common multi-word terms (e.g., "spider man", "star wars") if present.
        - Slide a window of size 2 and 3 over tokens; for each window:
            * Build a phrase string "t_i ... t_{i+w-1}".
            * If _is_common_phrase(phrase) → record phrase → list of tokens, and its positions range.
        - Return:
            * phrases: {phrase_str: [t1, t2, ...]}
            * phrase_positions: {phrase_str: [i, i+1, ...]}
        """
        # --------- TODO: Implement phrase mining over tokens ---------
        phrases = {}
        phrase_positions = {}
        
        for w in [3, 2]:
            if len(tokens) < w:
                continue
            for i in range(len(tokens) - w + 1):
                window_tokens = tokens[i:i+w]
                phrase_str = " ".join(window_tokens)
                if phrase_str not in phrases and self._is_common_phrase(phrase_str):
                    phrases[phrase_str] = window_tokens
                    phrase_positions[phrase_str] = list(range(i, i+w))
                    
        return phrases, phrase_positions

    # ---------------- STUDENT TODO ----------------
    def _is_common_phrase(self, phrase: str) -> bool:
        """
        Decide if a phrase is common enough to consider for boosting.

        Criteria (suggested):
        - All words exist in the index.
        - Phrase appears in at least 2 documents (intersection of doc sets per word).
        - Average positional distance between consecutive words across those docs
          (see _average_token_distance) is <= 6.0.
        """
        # --------- TODO: Implement phrase commonness check ---------
        words = phrase.split()
        doc_sets = []
        
        for w in words:
            if w not in self.inverted_index["index"]:
                return False
            docs = {p["doc_id"] for p in self.inverted_index["index"][w]["postings"]}
            doc_sets.append(docs)
            
        common_docs = set.intersection(*doc_sets) if doc_sets else set()
        
        if len(common_docs) < 2:
            return False
            
        avg_dist = self._average_token_distance(words, common_docs)
        return avg_dist <= 6.0

    # ---------------- STUDENT TODO ----------------
    def _average_token_distance(self, tokens: List[str], doc_ids: Set[str]) -> float:
        """
        Compute the average positional gap between consecutive tokens across docs.

        Steps:
        - For each doc in doc_ids:
            * Gather positions list for each token via _get_token_positions.
            * If all tokens present: for each consecutive pair, count positive gaps (pos2 - pos1 - 1) where pos2 > pos1.
        - Return mean gap over all counted pairs; if none, return float('inf').
        """
        # --------- TODO: Implement average gap calculation ---------
        total_gap = 0
        count = 0
        
        for doc_id in doc_ids:
            all_positions = []
            for token in tokens:
                all_positions.append(self._get_token_positions(doc_id, token))
            
            for i in range(len(tokens) - 1):
                pos1 = all_positions[i]
                pos2 = all_positions[i+1]
                
                for p1 in pos1:
                    for p2 in pos2:
                        if p2 > p1:
                            total_gap += (p2 - p1 - 1)
                            count += 1
                            
        return total_gap / count if count > 0 else float('inf')

    # ---------------- PROVIDED (keep) ----------------
    def _get_token_positions(self, doc_id: str, token: str) -> List[int]:
        """Get all positions of a token in a document."""
        idx = self.inverted_index["index"]
        if token not in idx:
            return []
        positions = []
        for posting in idx[token]["postings"]:
            if posting["doc_id"] == doc_id:
                positions.extend(posting["positions"])
        return sorted(positions)

    # ---------------- STUDENT TODO ----------------
    def _apply_phrase_boosts(self,
                             doc_scores: Dict[str, float],
                             phrase_terms: Dict[str, List[str]],
                             phrase_positions: Dict[str, List[int]],
                             detailed_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Apply boosts to documents containing phrase matches.

        Requirements:
        - For each phrase and each doc in doc_scores:
            * Check if all phrase tokens contributed to that doc (exist in detailed_scores[doc]).
            * Retrieve per-token positions via _get_token_positions.
            * If tokens form a flexible sequence (see _has_flexible_sequential_positions), add a boost:
                boost_amount = doc_scores[doc] * (self.phrase_boost - 1)
                boosted_scores[doc] += boost_amount
        - Return the boosted_scores dict.
        """
        # --------- TODO: Implement phrase boosting over matched docs ---------
        boosted_scores = dict(doc_scores)
        for phrase_str, window_tokens in phrase_terms.items():
            for doc in doc_scores:
                if all(t in detailed_scores[doc] for t in window_tokens):
                    all_positions = []
                    for t in window_tokens:
                        all_positions.append(self._get_token_positions(doc, t))
                    
                    if self._has_flexible_sequential_positions(all_positions):
                        boost_amount = doc_scores[doc] * (self.phrase_boost - 1.0)
                        boosted_scores[doc] += boost_amount
        return boosted_scores

    # ---------------- STUDENT TODO ----------------
    def _has_flexible_sequential_positions(self, all_positions: List[List[int]]) -> bool:
        """
        Decide if token positions form a near-sequential phrase.

        Rule of thumb:
        - Allow small gaps: each next token position must be within +3 of a previous position (next_pos > pos and next_pos - pos <= 3).
        - Progress token by token; if any step yields no candidates, return False.
        - If all steps succeed, return True.
        """
        # --------- TODO: Implement flexible phrase matching ---------
        if not all_positions:
            return False
            
        current_positions = all_positions[0]
        for i in range(1, len(all_positions)):
            next_positions = all_positions[i]
            new_positions = []
            for pos in current_positions:
                for next_pos in next_positions:
                    if next_pos > pos and next_pos - pos <= 3:
                        new_positions.append(next_pos)
            if not new_positions:
                return False
            current_positions = new_positions
        return True

    # ---------------- OPTIONAL (keep or ignore) ----------------
    def _has_sequential_positions_for_phrase(self, all_positions: List[List[int]]) -> bool:
        """Strict phrase check (exact adjacency). Provided for comparison; not required."""
        current_positions = all_positions[0]
        for i in range(1, len(all_positions)):
            next_positions = all_positions[i]
            new_positions = []
            for pos in current_positions:
                for next_pos in next_positions:
                    if next_pos == pos + 1:
                        new_positions.append(next_pos)
            if not new_positions:
                return False
            current_positions = new_positions
        return True

    # ---------------- PROVIDED (keep) ----------------
    def explain(self, query: str, doc_id: str) -> Dict[str, Any]:
        """Explain why a document ranked highly for a query (token & field contributions)."""
        field_weights, search_terms = self.query_parser.parse(query)
        query_tokens = search_terms.lower().split()

        explanation = {
            "query": query,
            "document_id": doc_id,
            "total_score": 0.0,
            "token_contributions": {},
            "field_contributions": defaultdict(float),
            "document_stats": {
                "length": getattr(self, "doc_lengths", {}).get(doc_id, 0),
                "avg_doc_length": getattr(self, "avg_doc_length", 1.0)
            }
        }

        for token in query_tokens:
            idx = self.inverted_index["index"]
            if token not in idx:
                continue
            token_data = idx[token]
            df = token_data["df"]
            idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)

            doc_contribution = 0.0
            for posting in token_data["postings"]:
                if posting["doc_id"] == doc_id:
                    tf = posting["tf"]
                    field = posting["field"]
                    doc_len = getattr(self, "doc_lengths", {}).get(doc_id, 1)
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / getattr(self, "avg_doc_length", 1.0)))
                    weight = field_weights.get(field, 1.0)
                    if weight <= 0:
                        weight = 0.01
                    token_score = weight * idf * (numerator / denominator)

                    doc_contribution += token_score
                    explanation["token_contributions"][token] = {
                        "score": token_score,
                        "tf": tf,
                        "idf": idf,
                        "field": field,
                        "field_weight": weight,
                        "normalized_tf": numerator / denominator
                    }
                    explanation["field_contributions"][field] += token_score

            explanation["total_score"] += doc_contribution

        explanation["field_contributions"] = dict(explanation["field_contributions"])
        logger.debug(f"Explanation generated for document {doc_id} with query '{query}'")
        return explanation

    # ---------------- PROVIDED (keep) ----------------
    def evaluate(self, test_queries: List[Tuple[str, List[str]]]) -> Dict[str, float]:
        """Compute Precision@5, Recall@5, and MRR over test_queries."""
        results = {"precision@5": [], "recall@5": [], "mrr": []}
        for query, relevant_docs in test_queries:
            search_results = self.search(query, k=5)
            retrieved_ids = [doc_id for doc_id, _, _ in search_results]
            relevant_retrieved = len(set(retrieved_ids) & set(relevant_docs))
            results["precision@5"].append(relevant_retrieved / 5)
            recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0
            results["recall@5"].append(recall)
            rr = 0
            for i, doc_id in enumerate(retrieved_ids, 1):
                if doc_id in relevant_docs:
                    rr = 1 / i
                    break
            results["mrr"].append(rr)
        return {m: (sum(v) / len(v) if v else 0.0) for m, v in results.items()}

def main() -> None:
    """
    Minimal driver for local testing.

    Usage:
        python search_engine.py "stars:1 title:0 summary:0.4 genres:0 drama prison"
    """
    try:
        search_engine = SearchEngine()
        # Simple smoke test
        q = "stars:1 title:0 summary:0.4 genres:0 drama prison"
        results = search_engine.search(q)
        print("\nTop results:")
        for i, (doc_id, score, _) in enumerate(results, 1):
            print(f"{i}. {doc_id}: {score:.4f}")
    except NotImplementedError as nie:
        logger.error(f"Unimplemented part of the assignment: {nie}")
        raise SystemExit(2)
    except Exception as e:
        logger.exception("Fatal error in search engine")
        print(f"\nSearch engine failed: {str(e)}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
