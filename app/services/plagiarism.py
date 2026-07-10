import json
import re
import math
from typing import List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from anthropic import Anthropic

from app.config import settings
from app.models import (
    ThesisVersion, CorpusDocument, CorpusSourceEnum,
    AnalysisSnapshot, Flag, FlagTypeEnum, SeverityEnum, RoleEnum
)

def get_closest_matches(db: Session, chunk_vector: List[float], this_thesis_id: str) -> List[Tuple[CorpusDocument, float]]:
    # 1. Fallback for SQLite testing (since SQLite lacks cosine_distance function)
    if db.bind.dialect.name == "sqlite":
        all_docs = db.query(CorpusDocument).all()
        matches = []
        for doc in all_docs:
            if doc.source_thesis_id == this_thesis_id:
                continue
            if doc.embedding and chunk_vector:
                # Resolve list/JSON string
                v1 = doc.embedding if isinstance(doc.embedding, list) else json.loads(doc.embedding)
                v2 = chunk_vector if isinstance(chunk_vector, list) else json.loads(chunk_vector)
                
                # Compute Cosine Similarity
                dot = sum(a * b for a, b in zip(v1, v2))
                norm1 = math.sqrt(sum(a * a for a in v1))
                norm2 = math.sqrt(sum(b * b for b in v2))
                similarity = dot / (norm1 * norm2) if (norm1 * norm2) > 0 else 0.0
                matches.append((doc, similarity))
        # Sort desc by similarity, limit to top 3
        return sorted(matches, key=lambda x: x[1], reverse=True)[:3]
        
    # 2. Postgres native pgvector search
    distance = CorpusDocument.embedding.cosine_distance(chunk_vector)
    query_res = db.query(
        CorpusDocument,
        (1.0 - distance).label("similarity")
    ).filter(
        # Exclude self thesis
        CorpusDocument.source_thesis_id != this_thesis_id
    ).order_by(
        distance.asc()
    ).limit(3).all()
    
    return [(row[0], float(row[1])) for row in query_res]

def evaluate_plagiarism_batch_llm(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Mock fallback for test mode
    api_key = settings.ANTHROPIC_API_KEY
    if not api_key or api_key == "mock_api_key_for_testing" or settings.ENV == "test":
        results = []
        for p in pairs:
            text_a = p["passage_a"].lower()
            text_b = p["passage_b"].lower()
            
            # Coincidental overlap mock
            if "coincidental" in text_a or "split" in text_a:
                results.append({
                    "pair_id": p["pair_id"],
                    "verdict": "coincidental_overlap",
                    "confidence": 0.9,
                    "evidence_excerpt": p["passage_a"][:40],
                    "reasoning": "Standard methodological phrasing."
                })
            # Cited mock
            elif "cite" in p["passage_a_context"].lower() or "cited" in text_a:
                results.append({
                    "pair_id": p["pair_id"],
                    "verdict": "properly_cited",
                    "confidence": 0.95,
                    "evidence_excerpt": p["passage_a"][:40],
                    "reasoning": "Work is properly cited in paragraph."
                })
            # Copied mock
            else:
                results.append({
                    "pair_id": p["pair_id"],
                    "verdict": "likely_copied",
                    "confidence": 0.85,
                    "evidence_excerpt": p["passage_a"][:40],
                    "reasoning": "Wording and structure are near-identical without attribution."
                })
        return results

    system_prompt = (
        "You are an academic integrity reviewer comparing two text passages for potential plagiarism.\n\n"
        "Passage A is from the thesis under review. Passage B is from a matched source (found via\n"
        "semantic similarity search — the match itself is not proof of wrongdoing, only a candidate).\n\n"
        "Determine:\n"
        "- \"likely_copied\": near-identical wording, structure, or argument with no attribution\n"
        "- \"likely_paraphrased\": same ideas/argument reworded, no attribution — this counts as\n"
        "  plagiarism even without matching wording\n"
        "- \"coincidental_overlap\": both passages discuss the same established concept using standard\n"
        "  field terminology — NOT plagiarism, this is expected in any two papers on a similar topic\n"
        "- \"properly_cited\": Passage A already attributes the idea/wording to Passage B's source (check\n"
        "  the surrounding text for citation markers) — NOT a violation\n\n"
        "Rules:\n"
        "- Standard terminology, common phrasings, and widely-known definitions are never plagiarism on\n"
        "  their own, even at high textual similarity.\n"
        "- If Passage A cites Passage B's source nearby, this is proper academic practice, not a\n"
        "  violation — check for this before flagging.\n"
        "- Quote the specific overlapping phrase(s) as evidence for your verdict.\n"
        "- Output strict JSON only."
    )

    user_content = f"Evaluate these candidate pairs:\n{json.dumps(pairs, indent=2)}\n\nReturn JSON array of results matching the template."

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4000,
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
        return json.loads(raw_res)
    except Exception as e:
        raise ValueError(f"Failed to parse Plagiarism LLM output: {e}. Raw: {raw_res}")

def run_plagiarism_review_pipeline(db: Session, version: ThesisVersion) -> AnalysisSnapshot:
    thesis = version.thesis
    raw_text = version.raw_text or ""
    
    # 1. Look up cached verdicts from the previous version of the same thesis
    verdict_cache = {}  # Map (chunk_content, corpus_doc_id) -> cached_result_dict
    if version.version_number > 1:
        prev_version = db.query(ThesisVersion).filter(
            ThesisVersion.thesis_id == version.thesis_id,
            ThesisVersion.version_number == version.version_number - 1
        ).first()
        if prev_version:
            prev_snapshot = db.query(AnalysisSnapshot).filter(
                AnalysisSnapshot.version_id == prev_version.id,
                AnalysisSnapshot.tool_type == "plagiarism"
            ).first()
            if prev_snapshot and prev_snapshot.scores and "verdicts" in prev_snapshot.scores:
                for v in prev_snapshot.scores["verdicts"]:
                    # Cache key: (chunk_content, matched_doc_id)
                    key = (v.get("passage_a"), v.get("corpus_document_id"))
                    verdict_cache[key] = v

    # 2. Iterate through Chunks
    candidate_pairs = []
    reused_verdicts = []
    
    for chunk in version.chunks:
        if not chunk.embedding:
            continue
            
        vector = chunk.embedding if isinstance(chunk.embedding, list) else json.loads(chunk.embedding)
        
        # Get top 3 semantic matches
        matches = get_closest_matches(db, vector, thesis.id)
        
        for doc, score in matches:
            # We filter by similarity threshold (discard < 0.80)
            if score < 0.80:
                continue
                
            # Check cache
            cache_key = (chunk.content, doc.id)
            if cache_key in verdict_cache:
                reused_verdicts.append(verdict_cache[cache_key])
                continue
                
            # Extract surrounding context (approx 200 chars before and after)
            idx = raw_text.find(chunk.content)
            if idx != -1:
                start_ctx = max(0, idx - 200)
                end_ctx = min(len(raw_text), idx + len(chunk.content) + 200)
                context = raw_text[start_ctx:end_ctx]
            else:
                context = chunk.content
                
            pair_id = f"chunk_{chunk.id}_vs_doc_{doc.id}"
            candidate_pairs.append({
                "pair_id": pair_id,
                "chunk_id": chunk.id,
                "corpus_document_id": doc.id,
                "source_type": doc.source_type,
                "source_thesis_id": doc.source_thesis_id,
                "source_title": doc.title,
                "passage_a": chunk.content,
                "passage_a_context": context,
                "passage_b": doc.chunk_text,
                "similarity_score": round(score, 3)
            })

    # 3. Batch LLM calls (groups of 8-10)
    llm_verdicts = []
    batch_size = 8
    for i in range(0, len(candidate_pairs), batch_size):
        batch = candidate_pairs[i:i+batch_size]
        # Format payload for the model (strip database keys before sending to LLM)
        llm_payload = []
        for b in batch:
            llm_payload.append({
                "pair_id": b["pair_id"],
                "passage_a": b["passage_a"],
                "passage_a_context": b["passage_a_context"],
                "passage_b": b["passage_b"],
                "similarity_score": b["similarity_score"]
            })
            
        results = evaluate_plagiarism_batch_llm(llm_payload)
        
        # Merge DB metadata back into LLM verdicts
        for r in results:
            # Find matching candidate from batch
            match = next((b for b in batch if b["pair_id"] == r["pair_id"]), None)
            if match:
                r.update({
                    "chunk_id": match["chunk_id"],
                    "corpus_document_id": match["corpus_document_id"],
                    "source_type": match["source_type"],
                    "source_thesis_id": match["source_thesis_id"],
                    "source_title": match["source_title"],
                    "passage_a": match["passage_a"],
                    "passage_b": match["passage_b"],
                    "similarity_score": match["similarity_score"]
                })
                llm_verdicts.append(r)

    # Combine new LLM judgments and cached/reused verdicts
    all_verdicts = llm_verdicts + reused_verdicts

    # 4. Generate Flags only for violations ("likely_copied" or "likely_paraphrased")
    plagiarism_flag_count = 0
    flags_to_create = []
    
    for v in all_verdicts:
        verdict_str = v.get("verdict")
        if verdict_str in ["likely_copied", "likely_paraphrased"]:
            # Format message with unmasked internal metadata (masked dynamically on student fetch)
            source_type = v.get("source_type")
            title = v.get("source_title") or "Unknown Title"
            thesis_id = v.get("source_thesis_id")
            
            if source_type == CorpusSourceEnum.internal_thesis:
                msg = f"Matches internal thesis [ID: {thesis_id}, Title: '{title}'] | Wording/structure overlaps without proper citation."
            else:
                msg = f"Matches external paper [Title: '{title}'] | Wording/structure overlaps without proper citation."
                
            evidence = v.get("evidence_excerpt", "")
            if not evidence or not str(evidence).strip():
                # Enforce non-empty evidence excerpt guardrail
                raise ValueError("Flag must carry a non-empty evidence excerpt.")
                
            severity = SeverityEnum.critical if verdict_str == "likely_copied" else SeverityEnum.moderate
            
            flag = Flag(
                type=FlagTypeEnum.plagiarism,
                severity=severity,
                message=msg,
                evidence_excerpt=evidence,
                page_ref=f"Similarity: {v.get('similarity_score')}"
            )
            flags_to_create.append(flag)
            plagiarism_flag_count += 1

    # 5. Write AnalysisSnapshot
    snapshot = AnalysisSnapshot(
        version_id=version.id,
        tool_type="plagiarism",
        scores={
            "flag_count": plagiarism_flag_count,
            "verdicts": all_verdicts
        }
    )
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)

    for flag in flags_to_create:
        flag.snapshot_id = snapshot.id
        db.add(flag)
        
    if flags_to_create:
        db.commit()

    return snapshot

def mask_plagiarism_flag_message(msg: str, user_role: str) -> str:
    # Privacy rule: mask title and student details for cross-student internal matches if role == student
    if user_role == RoleEnum.student:
        return re.sub(
            r"Matches internal thesis \[ID: [^,]+, Title: '[^']+'\]",
            "Matches another student's internal thesis [Details restricted to advisor]",
            msg
        )
    return msg
