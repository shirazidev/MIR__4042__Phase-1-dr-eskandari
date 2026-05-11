import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
import time
from typing import Dict, List, Any, Tuple, Optional

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_: Any):
        return iterable

# Configure production-grade logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("indexing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FieldedInvertedIndex:
    """
    Builds a fielded inverted index for movie search system.

    Why this implementation?
    - Stores field information for field-weighted search (title > summaries > stars > genres)
    - Includes positional information for phrase queries ("prison escape")
    - Memory-optimized for 1000+ movies with batch processing
    - No ranking logic - pure indexing (TF-IDF/BM25 belongs to search phase)

    Assignment note:
    Implement the TODO parts below. Do NOT change method names/signatures
    or the index schema: the autograder expects them.
    """

    def __init__(self,
                 processed_tokens_path: str = "fielded_processed_tokens.json",
                 inverted_index_path: str = "inverted_index.json",
                 batch_size: int = 200) -> None:
        """
        Initialize with paths and processing parameters.

        Args:
            processed_tokens_path: Path to fielded token data
            inverted_index_path: Destination for inverted index
            batch_size: Number of tokens per batch for memory optimization
        """
        self.processed_tokens_path = Path(processed_tokens_path)
        self.inverted_index_path = Path(inverted_index_path)
        self.inverted_index_path.parent.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.inverted_index: Dict[str, Dict[str, Any]] = {}

        # Only these 4 fields are indexed
        self.indexed_fields = ["title", "summaries", "genres", "stars"]

        # Track processing statistics
        self.stats = {
            "total_documents": 0,
            "total_tokens": 0,
            "unique_tokens": 0,
            "field_distribution": Counter(),
            "token_length_distribution": Counter()
        }

    def build(self) -> None:
        """Build the fielded inverted index from processed tokens."""
        start_time = time.time()
        logger.info(f"Starting fielded inverted index building from {self.processed_tokens_path}")

        # Load processed tokens with field information
        try:
            with open(self.processed_tokens_path, 'r', encoding='utf-8') as f:
                processed_data: Dict[str, Dict[str, List[str]]] = json.load(f)
            self.stats["total_documents"] = len(processed_data)
            logger.info(f"Loaded {len(processed_data)} processed movie entries")
        except Exception as e:
            logger.error(f"Failed to load processed tokens: {str(e)}")
            raise

        # Build index structure (STUDENT TODO inside this method)
        self._build_index_structure(processed_data)

        # Save index (provided)
        self._save_index()

        # Validation report (left fully implemented per instructions)
        self._generate_validation_report(processed_data)

        # Log completion
        elapsed = time.time() - start_time
        logger.info(f"Fielded inverted index built successfully in {elapsed:.2f} seconds")
        logger.info(f"Index contains {len(self.inverted_index)} unique tokens across {self.stats['total_documents']} documents")

    # ---------------- STUDENT TODO ----------------
    def _build_index_structure(self, processed_data: Dict[str, Dict[str, List[str]]]) -> None:
        """
        Build the fielded inverted index structure with positions.

        Requirements:
        - Iterate over each document (doc_id) and each indexed field.
        - Skip fields that are missing or empty.
        - Update self.stats:
            * self.stats["field_distribution"][field] += 1  (per doc if field has tokens)
            * self.stats["total_tokens"] += len(tokens)
            * self.stats["token_length_distribution"][len(token)] += 1 per token
        - For each token, enumerate positions and call:
            self._add_or_update_posting(token, doc_id, field, pos)
        - After processing all docs, set:
            self.stats["unique_tokens"] = len(self.inverted_index)
        """
        logger.info("Building fielded inverted index structure...")
        start_time = time.time()

        # --------- TODO: implement the loop over documents/fields/tokens ---------
        for doc_id, movie_data in processed_data.items():
            for field in self.indexed_fields:
                tokens = movie_data.get(field)
                if not tokens: continue
                self.stats["field_distribution"][field] += 1
                self.stats["total_tokens"] += len(tokens)
                for pos, token in enumerate(tokens):
                    self.stats["token_length_distribution"][len(token)] += 1
                    self._add_or_update_posting(token, doc_id, field, pos)
        
        self.stats["unique_tokens"] = len(self.inverted_index)

    # ---------------- STUDENT TODO ----------------
    def _add_or_update_posting(self, token: str, doc_id: str, field: str, position: int) -> None:
        """
        Add or update a posting for `token` in a specific `doc_id` and `field`.

        Required index schema per token:
        self.inverted_index[token] = {
            "df": <int>,                      # number of (doc_id, field) posting entries
            "postings": [
                {
                    "doc_id": <str>,
                    "tf": <int>,
                    "positions": [<int>, ...],
                    "field": <str>            # one of {'title','summaries','genres','stars'}
                },
                ...
            ]
        }

        Rules:
        - If this (doc_id, field) already exists in postings: increment tf and append `position`.
        - Else: append a new posting dict and increment df by 1.
        """
        # --------- TODO: implement posting maintenance logic ---------
        entry = self.inverted_index.setdefault(token, {"df": 0, "postings": []})
        found = False
        for posting in entry["postings"]:
            if posting["doc_id"] == doc_id and posting["field"] == field:
                posting["tf"] += 1
                posting["positions"].append(position)
                found = True
                break
        
        if not found:
            entry["postings"].append({
                "doc_id": doc_id,
                "tf": 1,
                "positions": [position],
                "field": field
            })
            entry["df"] += 1

    # ---------------- PROVIDED (keep) ----------------
    def _save_index(self) -> None:
        """Save the fielded inverted index to disk with memory efficiency."""
        logger.info(f"Saving index to {self.inverted_index_path}")
        start_time = time.time()

        try:
            # Save in chunks for memory efficiency (critical for 1000+ movies)
            def generate_json_chunks():
                yield '{\n  "metadata": {\n'
                yield f'    "indexed_fields": {json.dumps(self.indexed_fields)},\n'
                yield f'    "total_documents": {self.stats["total_documents"]},\n'
                yield f'    "total_tokens": {self.stats["total_tokens"]},\n'
                yield f'    "unique_tokens": {self.stats["unique_tokens"]},\n'
                yield f'    "index_date": "{time.strftime("%Y-%m-%d %H:%M:%S")}",\n'
                yield '    "field_distribution": {\n'

                # Format field distribution
                field_items = list(self.stats["field_distribution"].items())
                for i, (field, count) in enumerate(field_items):
                    yield f'      "{field}": {count}'
                    if i < len(field_items) - 1:
                        yield ',\n'
                    else:
                        yield '\n    },\n'

                # Format token length distribution
                yield '    "token_length_distribution": {\n'
                length_items = list(self.stats["token_length_distribution"].items())
                for i, (length, count) in enumerate(length_items):
                    yield f'      "{length}": {count}'
                    if i < len(length_items) - 1:
                        yield ',\n'
                    else:
                        yield '\n    }\n  },\n'

                # Index data
                yield '  "index": {\n'

                # Save tokens in chunks
                token_items = list(self.inverted_index.items())
                for i, (token, data) in enumerate(token_items):
                    yield f'    "{token}": {json.dumps(data)}'
                    if i < len(token_items) - 1:
                        yield ',\n'
                    else:
                        yield '\n  }\n}'

            with open(self.inverted_index_path, 'w', encoding='utf-8') as f:
                for chunk in generate_json_chunks():
                    f.write(chunk)

            elapsed = time.time() - start_time
            logger.info(f"Index saved successfully in {elapsed:.2f} seconds")

        except Exception as e:
            logger.error(f"Failed to save inverted index: {str(e)}")
            raise

    # ---------------- PROVIDED (keep) ----------------
    def _get_all_doc_ids(self) -> set:
        """Get all document IDs in the index."""
        doc_ids = set()
        for token_data in self.inverted_index.values():
            for posting in token_data["postings"]:
                doc_ids.add(posting["doc_id"])
        return doc_ids

    # ---------------- KEEP FULLY IMPLEMENTED ----------------
    def _generate_validation_report(self, processed_data: Dict[str, Dict[str, List[str]]]) -> None:
        """
        Generate comprehensive validation report for QA.

        Report includes:
        - Index statistics and metadata
        - Token frequency distribution
        - Sample token validation
        - Field coverage analysis
        - Performance metrics
        """
        report = [
            "===== FIELDED INVERTED INDEX VALIDATION REPORT =====",
            f"Generation timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total documents indexed: {self.stats['total_documents']}",
            f"Total tokens processed: {self.stats['total_tokens']}",
            f"Unique tokens: {self.stats['unique_tokens']}",
            "\n===== FIELD DISTRIBUTION ====="
        ]

        # Field coverage statistics
        total = sum(self.stats["field_distribution"].values()) or 1
        for field, count in self.stats["field_distribution"].items():
            report.append(f"  - {field}: {count} ({count/total*100:.1f}%)")

        # Token statistics
        report.extend([
            "\n===== TOKEN STATISTICS =====",
            f"Average tokens per document: {(self.stats['total_tokens']/self.stats['total_documents']):.1f}" if self.stats['total_documents'] else "Average tokens per document: 0.0",
            "\nToken length distribution:"
        ])

        length_items = sorted(self.stats["token_length_distribution"].items())
        for length, count in length_items[:15]:  # Show top 15 lengths
            report.append(f"  - Length {length}: {count}")
        if len(length_items) > 15:
            report.append("  - ... (truncated)")

        # Document frequency statistics
        doc_freqs = [item["df"] for item in self.inverted_index.values()]
        if doc_freqs:
            report.extend([
                "\n===== DOCUMENT FREQUENCY =====",
                f"Min DF: {min(doc_freqs)}",
                f"Max DF: {max(doc_freqs)}",
                f"Mean DF: {sum(doc_freqs)/len(doc_freqs):.2f}",
                f"Median DF: {sorted(doc_freqs)[len(doc_freqs)//2]}"
            ])

        # Sample token validation
        report.append("\n===== SAMPLE TOKEN VALIDATION =====")

        # Check critical tokens from knowledge base
        sample_tokens = [
            "morgan", "freeman", "prison", "redemption",
            "drama", "shawshank", "stars", "title"
        ]

        for token in sample_tokens:
            if token in self.inverted_index:
                token_data = self.inverted_index[token]
                report.append(f"\nToken: '{token}'")
                report.append(f"  - Document frequency: {token_data['df']}")
                report.append(f"  - Total occurrences: {sum(p['tf'] for p in token_data['postings'])}")
                report.append("  - Sample postings (max 3):")

                for i, posting in enumerate(token_data["postings"][:3]):
                    sample_positions = posting["positions"][:3]
                    positions_str = ", ".join(map(str, sample_positions))
                    report.append(f"    {i+1}. {posting['doc_id']} ({posting['field']}) - TF: {posting['tf']} - Positions: [{positions_str}{'...' if len(posting['positions']) > 3 else ''}]")
            else:
                report.append(f"\nToken: '{token}' - Not found in index")

        # Integrity checks
        report.append("\n===== INTEGRITY CHECKS =====")

        # Check if all documents are indexed
        indexed_docs = self._get_all_doc_ids()
        missing_docs = self.stats["total_documents"] - len(indexed_docs)
        if missing_docs == 0:
            report.append("✅ All documents are properly indexed")
        else:
            report.append(f"⚠️ {missing_docs} documents are missing from the index")

        # Check for inconsistent document frequencies
        df_errors = 0
        for token, data in self.inverted_index.items():
            if data["df"] != len(data["postings"]):
                df_errors += 1
        if df_errors == 0:
            report.append("✅ Document frequencies are consistent with posting list lengths")
        else:
            report.append(f"⚠️ {df_errors} tokens have inconsistent document frequencies")

        # Save report
        report_path = self.inverted_index_path.parent / "inverted_index_validation.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))

        logger.info(f"Validation report saved to {report_path}")

def main() -> None:
    """
    Execute inverted index building with proper error handling.

    Usage:
        python indexer.py
    """
    try:
        indexer = FieldedInvertedIndex(
            processed_tokens_path="fielded_processed_tokens.json",
            inverted_index_path="inverted_index.json",
            batch_size=200
        )
        indexer.build()
        print("\n✅ Fielded inverted index built successfully!")
        print(f"Total documents: {indexer.stats['total_documents']}")
        print(f"Unique tokens: {indexer.stats['unique_tokens']}")
        print(f"Index saved to: {indexer.inverted_index_path}")
        print(f"Validation report: {indexer.inverted_index_path.parent}/inverted_index_validation.txt")
        print("\nField distribution:")
        total_docs = indexer.stats['total_documents'] or 1
        for field, count in indexer.stats['field_distribution'].items():
            pct = (count/total_docs)*100 if total_docs else 0.0
            print(f"- {field}: {count} documents ({pct:.1f}%)")
    except NotImplementedError as nie:
        logger.error(f"Unimplemented part of the assignment: {nie}")
        raise SystemExit(2)
    except Exception as e:
        logger.exception("Fatal error in inverted index building pipeline")
        print(f"\n❌ Inverted index building failed: {str(e)}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
