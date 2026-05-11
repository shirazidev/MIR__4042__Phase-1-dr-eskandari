import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple
from pathlib import Path

# ---------- Pretty printing helpers ----------
GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; CYAN = "\033[96m"; BOLD = "\033[1m"; RESET = "\033[0m"

def banner(title: str) -> None:
    print(f"\n{BOLD}{CYAN}=== {title} ==={RESET}")

def ok(msg: str) -> None:
    print(f"{GREEN}✓{RESET} {msg}")

def warn(msg: str) -> None:
    print(f"{YELLOW}!{RESET} {msg}")

def fail(msg: str) -> None:
    print(f"{RED}✗{RESET} {msg}")

def outcome(passed: bool, name: str, reason: str = "") -> bool:
    if passed:
        ok(f"{name}: PASSED")
    else:
        fail(f"{name}: FAILED" + (f" — {reason}" if reason else ""))
    return passed

# ---------- Robust imports with explanations ----------
def safe_import(module: str, cls: str):
    try:
        mod = __import__(module, fromlist=[cls])
        return getattr(mod, cls)
    except Exception as e:
        return None

IMDbCrawler = safe_import("crawler", "IMDbCrawler")
DataProcessor = safe_import("preprocess", "DataProcessor")
FieldedInvertedIndex = safe_import("indexer", "FieldedInvertedIndex")
QueryParser = safe_import("query_parser", "QueryParser")
SearchEngine = safe_import("search_engine", "SearchEngine")

# ---------- Defaults / paths ----------
RAW_JSON = Path("IMDB_crawled.json")
TOKENS_JSON = Path("fielded_processed_tokens.json")
INDEX_JSON = Path("inverted_index.json")
ALLOWED_FIELDS = ["title", "summaries", "genres", "stars"]

# =====================================================
# A) Crawl-or-Load
# =====================================================
def grade_crawl_or_load(crawl: bool, limit: int, min_expected: int) -> Tuple[bool, str]:
    banner("A) Crawl or Load")
    print(crawl)
    print(IMDbCrawler)
    if RAW_JSON.exists() and not crawl:
        try:
            data = json.loads(RAW_JSON.read_text(encoding="utf-8"))
            if isinstance(data, list) and len(data) >= min_expected:
                ok(f"Loaded existing {RAW_JSON} with {len(data)} items")
                return True, ""
            warn(f"{RAW_JSON} exists but has too few items or wrong format; attempting to crawl")
        except Exception as e:
            warn(f"Could not read {RAW_JSON}: {e}; attempting to crawl")

    if not IMDbCrawler and crawl:
        return outcome(False, "Crawl", "imdb_crawler.IMDbCrawler not found"), "Missing IMDbCrawler"

    if crawl and IMDbCrawler:
        try:
            crawler = IMDbCrawler(crawling_threshold=limit, max_workers=10, min_interval=1.0, timeout=30, retries=6)
            crawler.start_crawling()
            crawler.write_to_file_as_json()
            data = json.loads(RAW_JSON.read_text(encoding="utf-8"))
            if isinstance(data, list) and len(data) >= min_expected:
                return outcome(True, "Crawl/Save"), ""
            return outcome(False, "Crawl/Save", f"Got {len(data) if isinstance(data, list) else 'malformed'} items"), "Crawl insufficient"
        except Exception as e:
            return outcome(False, "Crawl", str(e)), f"Crawl error: {e}"

    # Fallback: require existing file
    if RAW_JSON.exists():
        try:
            data = json.loads(RAW_JSON.read_text(encoding="utf-8"))
            if isinstance(data, list) and len(data) >= min_expected:
                return outcome(True, "Load"), ""
            else:
                return outcome(False, "Load", "Top-level not a list or too small"), "Load malformed"
        except Exception as e:
            return outcome(False, "Load", str(e)), f"Load error: {e}"
    else:
        return outcome(False, "Load", f"{RAW_JSON} not found"), "No data available"

# =====================================================
# B) Preprocess
# =====================================================
def sanity_check_processed(processed: Dict[str, Dict[str, List[str]]]) -> Tuple[bool, str]:
    if not isinstance(processed, dict) or not processed:
        return False, "Processed output is empty or not a dict"
    sample_items = 0
    for mid, fields in processed.items():
        if not isinstance(mid, str):
            return False, "Non-string movie id in processed output"
        if not isinstance(fields, dict):
            return False, f"Movie {mid} value is not a dict"
        # Only allowed fields
        for f in fields.keys():
            if f not in ALLOWED_FIELDS:
                return False, f"Unexpected field '{f}' in processed output"
        # Token lists are non-empty lists of strings
        for f, toks in fields.items():
            if not isinstance(toks, list):
                return False, f"Field '{f}' tokens for {mid} not a list"
            if not all(isinstance(t, str) for t in toks):
                return False, f"Non-string token in field '{f}' for {mid}"
        sample_items += 1
        if sample_items > 25:
            break
    return True, ""

def grade_preprocess(force: bool) -> Tuple[bool, str]:
    banner("B) Preprocess (DataProcessor)")
    if not DataProcessor:
        return outcome(False, "Import DataProcessor", "data_processor.DataProcessor not found"), "Missing DataProcessor"

    # If file exists and not forced, trust it, else re-run
    need_run = force or (not TOKENS_JSON.exists())
    try:
        if need_run:
            proc = DataProcessor(input_path=str(RAW_JSON), output_path=str(TOKENS_JSON), batch_size=200)
            processed = proc.process()
        else:
            processed = json.loads(TOKENS_JSON.read_text(encoding="utf-8"))

        passed, reason = sanity_check_processed(processed)
        return outcome(passed, "Preprocess Output Schema", reason), reason
    except Exception as e:
        return outcome(False, "Preprocess", str(e)), f"Preprocess error: {e}"

# =====================================================
# C) Indexer
# =====================================================
def sanity_check_index(idx: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(idx, dict) or "metadata" not in idx or "index" not in idx:
        return False, "Missing 'metadata' or 'index'"
    meta = idx["metadata"]; inv = idx["index"]
    if not isinstance(inv, dict) or not inv:
        return False, "Empty or malformed index data"
    # spot-check a handful of tokens
    n = 0
    for tok, data in inv.items():
        if not isinstance(tok, str):
            return False, "Non-string token key"
        if "df" not in data or "postings" not in data:
            return False, "Token entry missing df/postings"
        if not isinstance(data["postings"], list):
            return False, "Postings not a list"
        # df equals number of postings
        if data["df"] != len(data["postings"]):
            return False, f"DF mismatch for token '{tok}'"
        for p in data["postings"]:
            if not all(k in p for k in ("doc_id", "tf", "positions", "field")):
                return False, f"Posting missing keys for token '{tok}'"
            if p["field"] not in ALLOWED_FIELDS:
                return False, f"Posting field '{p['field']}' not allowed"
            if not isinstance(p["positions"], list) or not all(isinstance(i, int) for i in p["positions"]):
                return False, f"Positions malformed for token '{tok}'"
        n += 1
        if n > 50:
            break
    return True, ""

def grade_indexer(force: bool) -> Tuple[bool, str]:
    banner("C) Indexer (FieldedInvertedIndex)")
    if not FieldedInvertedIndex:
        return outcome(False, "Import FieldedInvertedIndex", "indexer.FieldedInvertedIndex not found"), "Missing FieldedInvertedIndex"

    need_run = force or (not INDEX_JSON.exists())
    try:
        if need_run:
            indexer = FieldedInvertedIndex(processed_tokens_path=str(TOKENS_JSON), inverted_index_path=str(INDEX_JSON), batch_size=200)
            indexer.build()
        idx = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
        passed, reason = sanity_check_index(idx)
        return outcome(passed, "Index Schema & Consistency", reason), reason
    except Exception as e:
        return outcome(False, "Indexer", str(e)), f"Indexer error: {e}"

# =====================================================
# D) Query Parser tests
# =====================================================
def grade_parser() -> Tuple[bool, str]:
    banner("D) Query Parser")
    if not QueryParser:
        return outcome(False, "Import QueryParser", "query_parser.QueryParser not found"), "Missing QueryParser"
    try:
        parser = QueryParser(inverted_index_path=str(INDEX_JSON))

        tests = [
            # weights only
            ("title:0 genres:1 stars:0 summaries:0 matrix", {"title":0.0,"genres":1.0,"summaries":0.0,"stars":0.0}, "matrix"),
            # alias handling (summary -> summaries)
            ("summary:0.4 drama", {"title":1.0,"genres":1.0,"summaries":0.4,"stars":1.0}, "drama"),
            # negative clamped to 0.0
            ("stars:-1 title:2 prison escape", {"title":2.0,"summaries":1.0,"genres":1.0,"stars":0.0}, "prison escape"),
            # unknown field -> treated as term
            ("director:1.0 tom hanks", {"title":1.0,"summaries":1.0,"genres":1.0,"stars":1.0}, "director:1.0 tom hanks"),
            # quotes removal preserves phrase
            ('stars:1 "lord of the rings"', {"title":1.0,"summaries":1.0,"genres":1.0,"stars":1.0}, "lord of the rings"),
        ]

        all_ok = True
        for i, (q, exp_w, exp_terms) in enumerate(tests, 1):
            w, t = parser.parse(q)
            # compare weights only for allowed fields; defaults for others
            weights_match = all(abs(w.get(f, 0)-exp_w.get(f,0)) < 1e-6 for f in ALLOWED_FIELDS)
            terms_match = (t == exp_terms)
            if weights_match and terms_match:
                ok(f"Parser test #{i} OK — '{q}'")
            else:
                all_ok = False
                if not weights_match:
                    fail(f"Parser test #{i} weights mismatch for '{q}'")
                    print("  expected:", {f:exp_w.get(f) for f in ALLOWED_FIELDS})
                    print("  actual  :", {f:w.get(f) for f in ALLOWED_FIELDS})
                if not terms_match:
                    fail(f"Parser test #{i} terms mismatch for '{q}'")
                    print("  expected:", exp_terms)
                    print("  actual  :", t)

        return outcome(all_ok, "Query Parser Suite"), "" if all_ok else "One or more parser tests failed"
    except Exception as e:
        return outcome(False, "Query Parser", str(e)), f"Parser error: {e}"

# =====================================================
# E) Search Engine evaluation
# =====================================================
def grade_search(k: int = 5) -> Tuple[bool, str]:
    banner("E) Search Engine (Precision@5, Recall@5, MRR)")

    if not SearchEngine:
        return outcome(False, "Import SearchEngine", "search_engine.SearchEngine not found"), "Missing SearchEngine"

    try:
        engine = SearchEngine(inverted_index_path=str(INDEX_JSON))

        # Queries targeting each field & mixes; ground-truth left empty for you to fill later.
        test_queries: List[Tuple[str, List[str]]] = [
            # Title-focused
            ("title:1 summaries:0 genres:0 stars:0 shawshank redemption", ["tt0111161"]),
            ("title:1 summaries:0 genres:0 stars:0 star wars", ["tt0121766", "tt0120915", "tt0121765", "tt0080684", "tt3748528",
                                                                "tt0076759", "tt0086190", "tt2527338", "tt2488496", "tt2527336", 
                                                                "tt0458290", "tt5118314", "tt8925010", "tt8924990", "tt13622982", 
                                                                "tt1643247", "tt20723374", "tt32019314", "tt12708542", "tt3778644"]),
            ("title:1 summaries:0 genres:0 stars:0 spider-man", ['tt16360004', 'tt0976192', 'tt9362722', 'tt4633694', 'tt0081938', 
                                                                 'tt0207120', 'tt1722512', 'tt9803682', 'tt10872600', 'tt27369002', 
                                                                 'tt2250912', 'tt1872181', 'tt0112175', 'tt6320628', 'tt0316654', 
                                                                 'tt0832446', 'tt0145487', 'tt0948470', 'tt0413300']),

            # Genres-focused
            ("genres:1 title:0 summaries:0 stars:0 drama crime adventure", ['tt0073343', 'tt36491653']),

            # Stars-focused (actors)
            ("stars:1 title:0 summaries:0 genres:0 tom hanks", ['tt29355505', 'tt0120689', 'tt0264464', 'tt1979376', 'tt0120815',
                                                                'tt0435761', 'tt0120363', 'tt0257044', 'tt0114709', 'tt0109830']),
            ("stars:1 title:0 summaries:0 genres:0 morgan freeman", ['tt0111161', 'tt0114369', 'tt0105695', 'tt0405159']),
            ("stars:1 title:0 summaries:0 genres:0 leonardo dicaprio", ['tt0993846', 'tt0264464', 'tt5537002', 'tt1663202', 'tt1853728',
                                                                        'tt11286314', 'tt0450259', 'tt1130884', 'tt0217505', 'tt1343092',
                                                                        'tt7131622', 'tt0407887', 'tt0338751', 'tt1375666', 'tt0120338']),

            # Summaries-focused (concepts / phrases)
            ("summaries:1 title:0 stars:0 genres:0 prison escape", ['tt0079116', 'tt0065063', 'tt0107808', 'tt0304141', 'tt0111161', 
                                                                    'tt0076584', 'tt0061512', 'tt1227537', 'tt7099566', 'tt0780504', 
                                                                    'tt0117500', 'tt0014358', 'tt0106519', 'tt0057115', 'tt1409024', 
                                                                    'tt0073486', 'tt0120586', 'tt0077416', 'tt0351283', 'tt0071771']),
            ("summaries:1 title:0 stars:0 genres:0 heist", ['tt27357183', 'tt0049406', 'tt0083190', 'tt1670345', 'tt0105236', 
                                                            'tt0478970', 'tt0240772', 'tt1323594', 'tt0454848', 'tt0113277', 
                                                            'tt3778644', 'tt0104431', 'tt0072890', 'tt2281587', 'tt0208092', 
                                                            'tt0050086', 'tt0074483', 'tt0065063', 'tt0095016', 'tt1375666']),
            ("summaries:1 title:0 stars:0 genres:0 revenge", ['tt0052618', 'tt0217505', 'tt0364569', 'tt0409221', 'tt0110413', 
                                                              'tt0064208', 'tt0413300', 'tt0096787', 'tt0068950', 'tt2911666', 
                                                              'tt0063032', 'tt0216165', 'tt1951266', 'tt0172495', 'tt0985694', 
                                                              'tt0101540', 'tt0458290', 'tt15239678', 'tt0257044', 'tt1431045', 
                                                              'tt0086383', 'tt1228705', 'tt6019206', 'tt1686804', 'tt0104815']),

            # Mixed weights
            ("title:0.7 genres:0 stars:0 summaries:0.3 lord of the rings", ['tt35882865', 'tt21811594', 'tt21811606', 'tt21811588', 'tt0077869', 
                                                                            'tt21822288', 'tt0120737', 'tt0167261', 'tt0167260']),
            ("title:0.5 summaries:0.5 genres:0 stars:0 back to the future", ['tt27458026', 'tt5124786', 'tt0096874', 'tt0388419', 'tt0099088'])
        ]

        # Print top-k for manual glance
        # threshold & k are configurable
        threshold = 0.80
        k = 5

        banner(f"Search pass/fail @k={k}, threshold={int(threshold*100)}%")

        pass_count = 0
        all_prec, all_rec = [], []

        for query, relevant in test_queries:
            res = engine.search(query, k=k)
            print(f"\nQuery: {query}")
            results = []
            for i, (doc_id, score, _) in enumerate(res, 1):
                results.append(doc_id)
                print(f"  {i:>2}. {doc_id}  {score:.4f}")

            passed, met = pass_recall_or_precision(results, relevant, k=k, threshold=threshold)
            msg = f"P@{k}={met['precision']:.3f}  R@{k}={met['recall']:.3f}  (|rel|={len(relevant)})"
            outcome(passed, "Query check", msg)

            pass_count += int(passed)
            all_prec.append(met["precision"])
            all_rec.append(met["recall"])

        # ---- macro summary over all queries ----
        if all_prec:
            p_macro = sum(all_prec) / len(all_prec)
            r_macro = sum(all_rec) / len(all_rec)
        else:
            p_macro = r_macro = 0.0

        macro_passed = (p_macro >= threshold) or (r_macro >= threshold)
        summary_msg = (
            f"macro P@{k}={p_macro:.3f}  macro R@{k}={r_macro:.3f}  "
            f"passed {pass_count}/{len(test_queries)} queries"
        )
        outcome(macro_passed, "Macro summary", summary_msg)

        # Compute metrics only when ground-truth provided
        # (Empty lists are treated as “labels pending” — metrics skipped gracefully.)
        labels_present = any(len(gt) > 0 for _, gt in test_queries)

        if labels_present:
            metrics = engine.evaluate(test_queries)
            ok(f"Metrics — Precision@{k}: {metrics['precision@5']:.4f}, Recall@{k}: {metrics['recall@5']:.4f}, MRR: {metrics['mrr']:.4f}")
            return outcome(True, "Search Evaluation"), ""
        else:
            warn("No ground-truth provided yet — metrics skipped. Fill the lists in test_queries to enable scoring.")
            return outcome(True, "Search Execution (no labels)"), ""

    except Exception as e:
        return outcome(False, "Search Engine", str(e)), f"Search error: {e}"

# =====================================================
# Micro-tests for each part (beyond end-to-end)
# =====================================================
def microtest_preprocess_tokens_sample() -> Tuple[bool, str]:
    """
    Light unit-style check: pick a handful of processed records and ensure:
    - at least one allowed field exists with non-empty tokens
    - no disallowed fields exist
    """
    try:
        data = json.loads(TOKENS_JSON.read_text(encoding="utf-8"))
        checked = 0
        for mid, fields in data.items():
            if not isinstance(fields, dict):
                return False, f"{mid} fields not a dict"
            if not any(f in fields and isinstance(fields[f], list) and fields[f] for f in ALLOWED_FIELDS):
                return False, f"{mid} has no non-empty token list among allowed fields"
            bad = [f for f in fields.keys() if f not in ALLOWED_FIELDS]
            if bad:
                return False, f"{mid} has unexpected fields: {bad}"
            checked += 1
            if checked >= 20:
                break
        return True, ""
    except Exception as e:
        return False, f"microtest_preprocess failed: {e}"

def microtest_index_df_consistency() -> Tuple[bool, str]:
    try:
        idx = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
        inv = idx["index"]
        n = 0
        for tok, data in inv.items():
            if data["df"] != len(data["postings"]):
                return False, f"DF mismatch for token '{tok}'"
            n += 1
            if n >= 200:
                break
        return True, ""
    except Exception as e:
        return False, f"microtest_index failed: {e}"

def microtest_parser_weights_only() -> Tuple[bool, str]:
    try:
        parser = QueryParser(inverted_index_path=str(INDEX_JSON))
        w, terms = parser.parse("title:0 genres:1 stars:0 summaries:0 noir thriller")
        if terms != "noir thriller":
            return False, "terms parsing unexpected"
        if not (abs(w.get("genres",0)-1.0) < 1e-6 and abs(w.get("title",0)) < 1e-6):
            return False, "weights parsing unexpected"
        return True, ""
    except Exception as e:
        return False, f"microtest_parser failed: {e}"
    
# =====================================================
# Metrics
# =====================================================
from typing import List, Tuple, Dict

def pass_recall_or_precision(
    results: List[str],
    relevant: List[str],
    k: int = None,
    threshold: float = 0.80
) -> Tuple[bool, Dict[str, float]]:
    """
    Returns (passed, {'precision': P, 'recall': R}).
    - passed=True if precision>=threshold OR recall>=threshold.
    - If k is given, uses top-k results; otherwise uses all.
    """
    rel = set(relevant)
    retrieved = results if (k is None) else results[:k]

    # Precision
    denom_p = max(1, len(retrieved))
    tp = sum(1 for d in retrieved if d in rel)
    precision = tp / denom_p

    # Recall
    denom_r = max(1, len(rel))
    recall = tp / denom_r

    passed = (precision >= threshold) or (recall >= threshold)
    return passed, {"precision": precision, "recall": recall}


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Autograder for 4-field IR pipeline")
    parser.add_argument("--crawl", action="store_true", help="Force crawling instead of loading existing JSON")
    parser.add_argument("--limit", type=int, default=1000, help="Crawling limit (default: 1000)")
    parser.add_argument("--min-count", type=int, default=900, help="Minimum movies expected to consider dataset usable")
    parser.add_argument("--force-preprocess", action="store_true", help="Force re-run preprocessing even if file exists")
    parser.add_argument("--force-index", action="store_true", help="Force re-run indexing even if file exists")
    parser.add_argument("--k", type=int, default=5, help="Top-k to print/evaluate (default: 5)")
    args = parser.parse_args()

    overall_ok = True
    reasons: List[str] = []

    # Crawl or load
    ok1, r1 = grade_crawl_or_load(args.crawl, args.limit, args.min_count)
    overall_ok &= bool(ok1); r1 and reasons.append(str(r1))

    # Preprocess
    ok2, r2 = grade_preprocess(force=args.force_preprocess)
    overall_ok &= bool(ok2); r2 and reasons.append(str(r2))

    # Micro-test preprocess
    banner("B.1) Preprocess Microtest")
    m1, mr1 = microtest_preprocess_tokens_sample()
    overall_ok &= m1; outcome(m1, "Tokens Sample Check", mr1); mr1 and reasons.append(mr1)

    # Index
    ok3, r3 = grade_indexer(force=args.force_index)
    overall_ok &= bool(ok3); r3 and reasons.append(str(r3))

    # Micro-test index
    banner("C.1) Index Microtest")
    m2, mr2 = microtest_index_df_consistency()
    overall_ok &= m2; outcome(m2, "DF Consistency (sample)", mr2); mr2 and reasons.append(mr2)

    # Parser
    ok4, r4 = grade_parser()
    overall_ok &= bool(ok4); r4 and reasons.append(str(r4))

    # Micro-test parser
    banner("D.1) Parser Microtest")
    m3, mr3 = microtest_parser_weights_only()
    overall_ok &= m3; outcome(m3, "Weights-only parsing", mr3); mr3 and reasons.append(mr3)

    # Search (evaluation)
    ok5, r5 = grade_search(k=args.k)
    overall_ok &= bool(ok5); r5 and reasons.append(str(r5))

    # Summary & exit code
    banner("AUTOGRADER SUMMARY")
    if overall_ok:
        ok("All stages completed")
        print(f"{GREEN}{BOLD}OVERALL: PASS{RESET}")
        sys.exit(0)
    else:
        fail("Some stages failed")
        print(f"{RED}{BOLD}OVERALL: FAIL{RESET}")
        # Show condensed reasons (unique, non-empty)
        seen = set()
        for r in reasons:
            if r and r not in seen:
                print("-", r)
                seen.add(r)
        sys.exit(1)

if __name__ == "__main__":
    main()
