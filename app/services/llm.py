import json
from typing import List, Dict, Any
from anthropic import Anthropic
from app.config import settings

class LLMService:
    def __init__(self):
        self.api_key = settings.ANTHROPIC_API_KEY
        if self.api_key and self.api_key != "mock_api_key_for_testing":
            self.client = Anthropic(api_key=self.api_key)
        else:
            self.client = None

    def verify_citations(self, field: str, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not claims:
            return []

        system_prompt = (
            "You are a citation verification specialist for academic theses. You will be given a batch of\n"
            "claims and their associated citations, along with any retrievable content from the cited source\n"
            "(abstract, or full text if available).\n\n"
            "For each claim-citation pair, determine:\n"
            "1. Does the citation exist in the bibliography? (given, not your job to re-check)\n"
            "2. Does the cited source's content actually support the specific claim made?\n"
            "   - \"yes\": the source directly supports the claim as stated\n"
            "   - \"partial\": the source is related but doesn't fully support the specific claim, or supports\n"
            "     a weaker/different version of it\n"
            "   - \"no\": the source contradicts the claim, or is about a substantially different topic\n"
            "   - \"unverifiable\": you don't have enough of the source's content to judge either way\n\n"
            "Rules:\n"
            "- Never mark \"no\" or \"partial\" without quoting the specific phrase in the claim that isn't\n"
            "  supported and explaining why.\n"
            "- If you only have an abstract and the claim is about a specific numeric result not mentioned\n"
            "  in the abstract, mark \"unverifiable\" — do not guess.\n"
            "- Do not fabricate content from the cited source. If source_snippet is empty, you cannot mark\n"
            "  \"yes\".\n"
            "- Output strict JSON only, no prose outside the JSON."
        )

        user_content = f"Verify the following {len(claims)} claim-citation pairs from a {field} thesis.\n\n"
        user_content += json.dumps(claims, indent=2) + "\n\n"
        user_content += (
            "Return JSON array, one object per input, in the same order:\n"
            "[\n"
            "  {\n"
            "    \"citation_key\": \"...\",\n"
            "    \"supports_claim\": \"yes\" | \"no\" | \"partial\" | \"unverifiable\",\n"
            "    \"confidence\": 0.0-1.0,\n"
            "    \"reasoning\": \"1-2 sentences, must quote the specific unsupported phrase if not 'yes'\"\n"
            "  }\n"
            "]"
        )

        if not self.client:
            # Return a stubbed/mocked response for development/testing if no real client is configured.
            results = []
            for claim in claims:
                snippet = claim.get("source_snippet", "")
                citation_key = claim.get("citation_key", "")
                if not snippet or not snippet.strip():
                    results.append({
                        "citation_key": citation_key,
                        "supports_claim": "unverifiable",
                        "confidence": 0.9,
                        "reasoning": "The source snippet is empty, so the claim cannot be verified."
                    })
                else:
                    # Default mock response support check
                    if "contradicts" in claim.get("claim_text", "").lower() or "wrong" in claim.get("claim_text", "").lower():
                        results.append({
                            "citation_key": citation_key,
                            "supports_claim": "no",
                            "confidence": 0.8,
                            "reasoning": f"The claim text contradicts the cited source snippet."
                        })
                    elif "partial" in claim.get("claim_text", "").lower() or "overclaim" in claim.get("claim_text", "").lower():
                        results.append({
                            "citation_key": citation_key,
                            "supports_claim": "partial",
                            "confidence": 0.85,
                            "reasoning": "The source supports translation quality, but not 'all NLP benchmarks'."
                        })
                    else:
                        results.append({
                            "citation_key": citation_key,
                            "supports_claim": "yes",
                            "confidence": 0.95,
                            "reasoning": "The source snippet supports the claim."
                        })
            return results

        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_content}
            ],
            temperature=0.0
        )
        
        raw_text = response.content[0].text.strip()
        try:
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()
            return json.loads(raw_text)
        except Exception as e:
            raise ValueError(f"Failed to parse Claude response as JSON: {e}. Raw response: {raw_text}")

llm_service = LLMService()
