import re
import httpx
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from app.config import settings
from app.models import (
    ThesisVersion, CitationRecord, AnalysisSnapshot, Flag,
    FlagTypeEnum, SeverityEnum, SupportEnum
)
from app.services.llm import llm_service

# Regex definitions
NUMERIC_RE = re.compile(r'\[(\d+(?:\s*,\s*\d+)*)\]')
AUTHOR_DATE_RE = re.compile(r'\(([A-Za-z]+(?:\s+et\s+al\.)?,?\s+\d{4})\)')
DOI_RE = re.compile(r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b', re.IGNORECASE)

def split_into_sentences(text: str) -> List[str]:
    # Split sentences on periods, questions, or exclamation marks followed by spaces
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def parse_bibliography(raw_text: str) -> Dict[str, str]:
    bib_headers = [
        r'\bReferences\b',
        r'\bBibliography\b',
        r'\bWorks\s+Cited\b'
    ]
    
    split_idx = -1
    for header in bib_headers:
        matches = list(re.finditer(header, raw_text, re.IGNORECASE))
        if matches:
            split_idx = matches[-1].start()
            break
            
    if split_idx == -1:
        return {}
        
    bib_text = raw_text[split_idx:]
    lines = bib_text.split('\n')
    bib_map = {}
    
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        
        # Check standard numbering styles: [1] or 1.
        num_prefix = re.match(r'^\[(\d+)\]\s*(.*)', line_str)
        if num_prefix:
            key = f"[{num_prefix.group(1)}]"
            bib_map[key] = num_prefix.group(2).strip()
            continue
            
        num_dot_prefix = re.match(r'^(\d+)\.\s*(.*)', line_str)
        if num_dot_prefix:
            key = f"[{num_dot_prefix.group(1)}]"
            bib_map[key] = num_dot_prefix.group(2).strip()
            continue
            
        # Check Harvard/Author-Date bibliography style: e.g. "Smith, J. (2023)..."
        author_date = re.match(r'^([A-Za-z]+)(?:,\s+[A-Z]\.)*\s+\((\d{4})\)', line_str)
        if author_date:
            name = author_date.group(1)
            year = author_date.group(2)
            bib_map[f"({name}, {year})"] = line_str
            bib_map[f"({name} {year})"] = line_str
            bib_map[f"{name} {year}"] = line_str
            
    return bib_map

async def fetch_crossref_metadata(doi: str) -> Optional[str]:
    url = f"https://api.crossref.org/works/{doi}"
    headers = {
        "User-Agent": f"ThesisRAGPlatform/1.0 (mailto:{settings.CROSSREF_POLITE_EMAIL})"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                message = data.get("message", {})
                
                # Check abstract
                abstract = message.get("abstract", "")
                if abstract:
                    # Strip JATS XML tags if present
                    abstract = re.sub(r'<[^>]+>', '', abstract).strip()
                    return abstract
                
                # Fallback to title and container title
                title_list = message.get("title", [])
                title = title_list[0] if title_list else "Unknown Title"
                container = message.get("container-title", [])
                container_title = container[0] if container else "Unknown Source"
                return f"Title: {title}. Container: {container_title}."
        except Exception:
            return None
    return None

def find_cached_source_snippet(db: Session, doi: str) -> Optional[str]:
    record = db.query(CitationRecord).filter(
        CitationRecord.doi == doi,
        CitationRecord.source_snippet.isnot(None),
        CitationRecord.source_snippet != ""
    ).first()
    return record.source_snippet if record else None

async def run_citation_verification_pipeline(db: Session, version: ThesisVersion) -> List[CitationRecord]:
    # 1. Parse bibliography
    bib_map = parse_bibliography(version.raw_text)
    
    # 2. Split body from bibliography to avoid scanning bibliography references as in-text citations
    bib_headers = [
        r'\bReferences\b',
        r'\bBibliography\b',
        r'\bWorks\s+Cited\b'
    ]
    body_text = version.raw_text
    for header in bib_headers:
        matches = list(re.finditer(header, version.raw_text, re.IGNORECASE))
        if matches:
            body_text = version.raw_text[:matches[-1].start()]
            break

    # 3. Setup incremental analysis helpers
    paragraphs = body_text.split('\n\n')
    prev_version = None
    if version.version_number > 1:
        prev_version = db.query(ThesisVersion).filter(
            ThesisVersion.thesis_id == version.thesis_id,
            ThesisVersion.version_number == version.version_number - 1
        ).first()

    changed_paragraphs = []
    is_incremental = False
    if prev_version and version.diff_from_prev:
        changed_paragraphs = version.diff_from_prev.get("changed_paragraphs", [])
        is_incremental = True

    prev_records = {}
    if prev_version:
        records = db.query(CitationRecord).filter(CitationRecord.version_id == prev_version.id).all()
        for r in records:
            prev_records[(r.citation_key, r.claim_text)] = r

    citation_records_to_create = []
    claims_to_verify = []

    for p_idx, p in enumerate(paragraphs):
        p_text = p.strip()
        if not p_text:
            continue
            
        paragraph_changed = True
        if is_incremental:
            paragraph_changed = any(cp in p_text or p_text in cp for cp in changed_paragraphs)
            
        sentences = split_into_sentences(p_text)
        for s in sentences:
            citations_in_sentence = []
            
            # Find all numeric references e.g. [1, 2]
            for m in NUMERIC_RE.finditer(s):
                keys_str = m.group(1)
                for k in keys_str.split(','):
                    citations_in_sentence.append(f"[{k.strip()}]")
                    
            # Find all author-date references e.g. (Smith 2023)
            for m in AUTHOR_DATE_RE.finditer(s):
                citations_in_sentence.append(m.group(0))
                
            for cite_key in citations_in_sentence:
                # Resolve bibliography match
                bib_entry = bib_map.get(cite_key)
                if not bib_entry:
                    normalized_key = cite_key.strip("()")
                    bib_entry = bib_map.get(normalized_key)
                
                exists_in_bib = bib_entry is not None
                format_ok = exists_in_bib  # Format is considered ok if it matched regex and is in references
                
                doi = None
                if bib_entry:
                    doi_match = DOI_RE.search(bib_entry)
                    if doi_match:
                        doi = doi_match.group(1)
                
                claim_loc = f"Para {p_idx + 1}"
                
                # Check if we can reuse previous record (incremental)
                if is_incremental and not paragraph_changed:
                    prev_r = prev_records.get((cite_key, s))
                    if prev_r:
                        new_r = CitationRecord(
                            version_id=version.id,
                            citation_key=cite_key,
                            claim_text=s,
                            claim_location=claim_loc,
                            exists_in_bib=prev_r.exists_in_bib,
                            doi=prev_r.doi,
                            format_ok=prev_r.format_ok,
                            supports_claim=prev_r.supports_claim,
                            confidence=prev_r.confidence,
                            source_snippet=prev_r.source_snippet,
                            reasoning=prev_r.reasoning
                        )
                        citation_records_to_create.append(new_r)
                        continue

                # Fetch/Cache source snippet
                source_snippet = None
                if doi:
                    source_snippet = find_cached_source_snippet(db, doi)
                    if not source_snippet:
                        source_snippet = await fetch_crossref_metadata(doi)
                
                new_r = CitationRecord(
                    version_id=version.id,
                    citation_key=cite_key,
                    claim_text=s,
                    claim_location=claim_loc,
                    exists_in_bib=exists_in_bib,
                    doi=doi,
                    format_ok=format_ok,
                    source_snippet=source_snippet,
                    supports_claim=SupportEnum.unverifiable,
                    confidence=0.0
                )
                citation_records_to_create.append(new_r)
                
                # If it exists in bib, add to batch for LLM verification
                if exists_in_bib:
                    claims_to_verify.append({
                        "record_idx": len(citation_records_to_create) - 1,
                        "citation_key": cite_key,
                        "claim_text": s,
                        "claim_location": claim_loc,
                        "source_snippet": source_snippet or ""
                    })
                else:
                    # If it doesn't exist in bib, it's immediately unverifiable
                    new_r.supports_claim = SupportEnum.unverifiable

    # 3. Batch LLM verification
    batch_size = 12
    for i in range(0, len(claims_to_verify), batch_size):
        batch = claims_to_verify[i:i+batch_size]
        llm_inputs = [{
            "citation_key": item["citation_key"],
            "claim_text": item["claim_text"],
            "claim_location": item["claim_location"],
            "source_snippet": item["source_snippet"]
        } for item in batch]
        
        field = version.thesis.field or "General"
        verifications = llm_service.verify_citations(field, llm_inputs)
        
        for v_res, item in zip(verifications, batch):
            rec = citation_records_to_create[item["record_idx"]]
            
            supports = v_res.get("supports_claim", "unverifiable")
            if supports not in ["yes", "no", "partial", "unverifiable"]:
                supports = "unverifiable"
                
            # Non-negotiable rule: If source_snippet is empty, must be unverifiable
            if not rec.source_snippet or not rec.source_snippet.strip():
                supports = "unverifiable"
                
            rec.supports_claim = SupportEnum(supports)
            rec.confidence = float(v_res.get("confidence", 0.0))
            rec.reasoning = v_res.get("reasoning", "")

    # Save all citation records
    for r in citation_records_to_create:
        db.add(r)
    db.commit()

    # 4. Create AnalysisSnapshot & Flags
    total_citations = len(citation_records_to_create)
    total_valid = sum(1 for r in citation_records_to_create if r.supports_claim == SupportEnum.yes)
    score = int((total_valid / total_citations) * 100) if total_citations > 0 else 100
    
    snapshot = AnalysisSnapshot(
        version_id=version.id,
        tool_type="citation_verification",
        scores={
            "overall": score,
            "total_citations": total_citations,
            "valid_citations": total_valid
        }
    )
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)

    for r in citation_records_to_create:
        flag_message = None
        severity = SeverityEnum.low
        evidence = None

        if not r.exists_in_bib:
            flag_message = f"Citation '{r.citation_key}' does not exist in the bibliography."
            severity = SeverityEnum.critical
            evidence = f"In-text citation: '{r.claim_text}' could not be matched to any reference."
        elif r.supports_claim == SupportEnum.no:
            flag_message = f"Citation '{r.citation_key}' does not support the claim: '{r.claim_text}'"
            severity = SeverityEnum.critical
            evidence = f"Claim: '{r.claim_text}'\nCited snippet: '{r.source_snippet or 'None'}'"
        elif r.supports_claim == SupportEnum.partial:
            flag_message = f"Citation '{r.citation_key}' only partially supports the claim: '{r.claim_text}'"
            severity = SeverityEnum.moderate
            evidence = f"Claim: '{r.claim_text}'\nCited snippet: '{r.source_snippet or 'None'}'"

        if flag_message:
            # Enforce non-null evidence excerpt at application layer
            if not evidence or not evidence.strip():
                evidence = f"Citation Issue: {flag_message}"
                
            flag = Flag(
                snapshot_id=snapshot.id,
                type=FlagTypeEnum.citation,
                severity=severity,
                message=flag_message,
                evidence_excerpt=evidence,
                page_ref=r.claim_location
            )
            db.add(flag)
            
    db.commit()
    return citation_records_to_create
