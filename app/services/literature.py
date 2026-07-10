import json
import math
from typing import List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from anthropic import Anthropic

from app.config import settings
from app.models import ThesisVersion, CorpusDocument, CorpusSourceEnum
from app.services.novelty import is_contribution_chunk

def get_vector_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2) if (norm1 * norm2) > 0 else 0.0

def run_literature_clustering_and_synthesis(
    db: Session,
    version: ThesisVersion
) -> Dict[str, Any]:
    thesis = version.thesis
    
    # 1. Retrieve all external papers from CorpusDocument cache
    external_docs = db.query(CorpusDocument).filter(
        CorpusDocument.source_type == CorpusSourceEnum.external_paper
    ).all()

    if not external_docs:
        return {"clusters": []}

    # 2. Identify contribution-claim chunks from the current thesis version
    contrib_chunks = []
    for chunk in version.chunks:
        if chunk.embedding and is_contribution_chunk(chunk, version.section_map or {}):
            contrib_chunks.append(chunk)
            
    # Fallback to all chunks if no contribution claim chunks match the keywords
    if not contrib_chunks:
        contrib_chunks = [c for c in version.chunks if c.embedding]

    if not contrib_chunks:
        return {"clusters": []}

    # 3. Calculate max similarity of each external paper to the thesis's chunks
    candidate_pool: List[Tuple[CorpusDocument, List[float], float]] = []
    for doc in external_docs:
        if not doc.embedding:
            continue
            
        doc_vec = doc.embedding if isinstance(doc.embedding, list) else json.loads(doc.embedding)
        
        max_sim = -1.0
        for chunk in contrib_chunks:
            chunk_vec = chunk.embedding if isinstance(chunk.embedding, list) else json.loads(chunk.embedding)
            sim = get_vector_similarity(doc_vec, chunk_vec)
            if sim > max_sim:
                max_sim = sim
                
        candidate_pool.append((doc, doc_vec, max_sim))

    # Sort candidates by similarity to thesis chunks descending, and cap at 20 papers
    candidate_pool = sorted(candidate_pool, key=lambda x: x[2], reverse=True)[:20]

    # 4. Greedy Clustering (Seed-to-Candidate only, consumed pool)
    clusters = []
    while candidate_pool:
        # Seed is the paper closest to the thesis chunks
        seed_doc, seed_vec, seed_score = candidate_pool.pop(0)
        cluster_docs = [seed_doc]
        
        # Look for matches within 0.15 cosine distance (similarity >= 0.85) to the seed paper
        matches = []
        for cand in candidate_pool:
            cand_doc, cand_vec, cand_score = cand
            sim = get_vector_similarity(seed_vec, cand_vec)
            if sim >= 0.85:
                matches.append(cand)
                
        # Consume match documents from pool
        for m in matches:
            cluster_docs.append(m[0])
            candidate_pool.remove(m)
            
        clusters.append(cluster_docs)

    # 5. LLM Synthesis stage
    final_clusters = []
    api_key = settings.ANTHROPIC_API_KEY
    
    for idx, cluster in enumerate(clusters):
        theme_label = f"Cluster theme {idx + 1}"
        papers_payload = []
        for p in cluster:
            papers_payload.append({
                "title": p.title or "Unknown Title",
                "abstract": p.chunk_text or "",
                "paperId": p.external_id or "unknown"
            })

        # Mock mode fallback
        if not api_key or api_key == "mock_api_key_for_testing" or settings.ENV == "test":
            title_list = [p["title"] for p in papers_payload]
            draft_paragraph = (
                f"The papers in this cluster focus on transformer pruning optimizations. "
                f"Specifically, works like '{title_list[0]}' introduce weight pruning heuristics. "
                "These approaches collectively build toward reducing attention head complexity during inference, "
                "though they diverge on whether layers are pruned statically or dynamically. "
                "Students can build on these baselines by proposing structured head-elimination algorithms."
            )
            final_clusters.append({
                "theme_label": f"Attention pruning optimization (Theme {idx + 1})",
                "papers": [{"title": p.title, "external_id": p.external_id, "abstract": p.chunk_text} for p in cluster],
                "synthesis_paragraph": draft_paragraph
            })
            continue

        system_prompt = (
            "You are helping a student draft a literature review section by synthesizing a cluster of related papers.\n\n"
            "You will receive a group of paper abstracts that share a research theme. Write a short synthesis paragraph (100-150 words) that:\n"
            "- States the shared theme/approach across these papers\n"
            "- Notes where they agree or build on each other\n"
            "- Notes any meaningful disagreement or divergent approach among them, if present\n"
            "- Ends with a natural transition sentence a student could use to introduce this cluster in their own related-work section\n\n"
            "Rules:\n"
            "- This is a DRAFT for the student to edit and cite properly themselves — do not write it as if it's the final thesis text with confident unattributed claims.\n"
            "- Do not invent findings not present in the abstracts provided.\n"
            "- Every specific claim about a paper's findings must be traceable to that paper's abstract.\n"
            "- Output strict JSON only."
        )

        user_content = (
            f"Cluster theme (working label): \"{theme_label}\"\n"
            f"Papers:\n"
            f"{json.dumps(papers_payload, indent=2)}\n\n"
            "Return JSON:\n"
            "{\n"
            "  \"theme_label\": \"refined theme name\",\n"
            "  \"synthesis_paragraph\": \"...\",\n"
            "  \"paper_citations\": [\"Author et al. 2023\", ...]\n"
            "}"
        )

        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.0
        )

        raw_res = response.content[0].text.strip()
        try:
            if "```json" in raw_res:
                raw_res = raw_res.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_res:
                raw_res = raw_res.split("```")[1].split("```")[0].strip()
            synthesis_json = json.loads(raw_res)
            
            final_clusters.append({
                "theme_label": synthesis_json.get("theme_label", theme_label),
                "papers": [{"title": p.title, "external_id": p.external_id, "abstract": p.chunk_text} for p in cluster],
                "synthesis_paragraph": synthesis_json.get("synthesis_paragraph", "")
            })
        except Exception as e:
            # Fallback on parse failure
            final_clusters.append({
                "theme_label": theme_label,
                "papers": [{"title": p.title, "external_id": p.external_id, "abstract": p.chunk_text} for p in cluster],
                "synthesis_paragraph": f"Synthesis draft for cluster papers: {', '.join([p.title for p in cluster])}."
            })

    return {"clusters": final_clusters}
