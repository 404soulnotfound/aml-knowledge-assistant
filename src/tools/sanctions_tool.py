import json
from pathlib import Path
from typing import Dict, Any, List
from difflib import SequenceMatcher
from ..config import settings


def _string_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def load_sanctions_data() -> Dict[str, Any]:
    file_path = settings.SANCTIONS_FILE
    if not file_path.exists():
        return {"sanctions_list": [], "pep_list": [], "high_risk_jurisdictions": {"black_list": [], "grey_list": []}}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def search_sanctions_and_pep(name: str, threshold: float = 0.65) -> Dict[str, Any]:
    """
    Screens an individual or entity name against international sanctions (OFAC, UN, EU)
    and Politically Exposed Persons (PEP) registries.
    
    Args:
        name: Name of the individual or legal entity to screen.
        threshold: Fuzzy similarity threshold (0.0 to 1.0) for matching.
        
    Returns:
        Dictionary containing matched sanction records, PEP records, and screening verdict.
    """
    db = load_sanctions_data()
    clean_query = name.lower().strip()
    
    matched_sanctions: List[Dict[str, Any]] = []
    for item in db.get("sanctions_list", []):
        score = _string_similarity(clean_query, item["name"])
        matched_alias = None
        for alias in item.get("aliases", []):
            alias_score = _string_similarity(clean_query, alias)
            if alias_score > score:
                score = alias_score
                matched_alias = alias
                
        # Substring exact check
        if clean_query in item["name"].lower() or any(clean_query in a.lower() for a in item.get("aliases", [])):
            score = max(score, 0.90)
            
        if score >= threshold:
            matched_sanctions.append({
                **item,
                "match_score": round(score, 3),
                "matched_on": matched_alias if matched_alias and score > _string_similarity(clean_query, item["name"]) else item["name"]
            })
            
    matched_peps: List[Dict[str, Any]] = []
    for pep in db.get("pep_list", []):
        score = _string_similarity(clean_query, pep["name"])
        if clean_query in pep["name"].lower():
            score = max(score, 0.90)
            
        for family_member in pep.get("family_members_or_associates", []):
            fam_score = _string_similarity(clean_query, family_member)
            if fam_score >= threshold:
                score = max(score, fam_score * 0.9)
                
        if score >= threshold:
            matched_peps.append({
                **pep,
                "match_score": round(score, 3)
            })
            
    has_sanction = len(matched_sanctions) > 0
    has_pep = len(matched_peps) > 0
    
    if has_sanction:
        status = "CRITICAL_BLOCKED"
        recommendation = "Immediate asset freeze and prohibition of transaction per statutory sanctions regime. Mandatory filing of blocked property report."
    elif has_pep:
        status = "HIGH_RISK_PEP"
        recommendation = "Escalate for Senior Management Approval. Conduct Enhanced Due Diligence (EDD) to verify source of wealth and funds."
    else:
        status = "CLEAR"
        recommendation = "No adverse screening matches detected against watchlists. Proceed with standard customer due diligence (CDD)."
        
    return {
        "query": name,
        "status": status,
        "recommendation": recommendation,
        "sanctions_matches": matched_sanctions,
        "pep_matches": matched_peps,
        "total_hits": len(matched_sanctions) + len(matched_peps)
    }
