import datetime
import httpx
from sqlalchemy.orm import Session
from typing import List, Dict, Any

from app.config import settings
from app.models import CorpusDocument, CorpusSourceEnum, SemanticScholarSearch
from app.services.ingestion import get_embeddings

def query_semanticscholar(query_text: str) -> List[Dict[str, Any]]:
    # Mock fallback for test environment
    if settings.ENV == "test" or settings.ANTHROPIC_API_KEY == "mock_api_key_for_testing":
        return [
            {
                "paperId": "mock_paper_1",
                "title": "Structured Pruning of Attention Heads for Efficient Transformers",
                "abstract": "We introduce a method for pruning attention heads with low importance scores during inference, reducing FLOPs by up to 40% with minimal accuracy loss."
            },
            {
                "paperId": "mock_paper_2",
                "title": "Attention Head Pruning Baselines",
                "abstract": "Pruning attention heads using magnitude or weight thresholds is a standard baseline to improve inferencing speeds in larger models."
            }
        ]

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query_text,
        "limit": 5,
        "fields": "title,abstract,paperId"
    }
    
    try:
        response = httpx.get(url, params=params, timeout=15.0)
        if response.status_code == 200:
            data = response.json()
            return data.get("data", [])
        else:
            # Fallback gracefully instead of failing entire request if SS is down
            print(f"Semantic Scholar API error: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"Semantic Scholar request failed: {e}")
        return []

def search_and_cache_external_papers(db: Session, query_text: str):
    query_clean = query_text.strip().lower()
    if not query_clean:
        return

    # 1. Check if search query was executed in the last 30 days
    thirty_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=30)
    existing_search = db.query(SemanticScholarSearch).filter(
        SemanticScholarSearch.query_text == query_clean,
        SemanticScholarSearch.fetched_at >= thirty_days_ago
    ).first()

    if existing_search:
        # Already fetched and cached recently, skip external search to avoid API hit
        print(f"Skipping Semantic Scholar API search for query '{query_clean}' (cached).")
        return

    # 2. Fetch papers from Semantic Scholar API
    print(f"Querying Semantic Scholar API for: '{query_clean}'")
    papers = query_semanticscholar(query_clean)

    # 3. Cache new papers into CorpusDocument
    new_docs_to_embed = []
    for paper in papers:
        paper_id = paper.get("paperId")
        if not paper_id:
            continue
            
        title = paper.get("title")
        abstract = paper.get("abstract") or ""
        
        # Check if already cached by external_id
        cached = db.query(CorpusDocument).filter(
            CorpusDocument.external_id == paper_id,
            CorpusDocument.source_type == CorpusSourceEnum.external_paper
        ).first()
        
        if not cached and abstract:
            new_docs_to_embed.append({
                "external_id": paper_id,
                "title": title,
                "abstract": abstract
            })

    # 4. Generate embeddings for any newly found papers in batch
    if new_docs_to_embed:
        abstracts = [doc["abstract"] for doc in new_docs_to_embed]
        embeddings = get_embeddings(abstracts)
        
        for idx, doc in enumerate(new_docs_to_embed):
            vector = embeddings[idx] if idx < len(embeddings) else None
            corpus_doc = CorpusDocument(
                source_type=CorpusSourceEnum.external_paper,
                external_id=doc["external_id"],
                title=doc["title"],
                chunk_text=doc["abstract"],
                embedding=vector
            )
            db.add(corpus_doc)
        db.commit()

    # 5. Log this search query
    search_log = db.query(SemanticScholarSearch).filter(
        SemanticScholarSearch.query_text == query_clean
    ).first()
    if search_log:
        search_log.fetched_at = datetime.datetime.utcnow()
    else:
        search_log = SemanticScholarSearch(
            query_text=query_clean
        )
        db.add(search_log)
        
    db.commit()
