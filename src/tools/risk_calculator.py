from typing import Dict, Any, List
from ..config import settings
from .sanctions_tool import load_sanctions_data


def calculate_transaction_risk(
    amount: float,
    currency: str = "USD",
    country: str = "US",
    entity_type: str = "Individual",
    payment_method: str = "Wire Transfer",
    counterparty_country: str = ""
) -> Dict[str, Any]:
    """
    Evaluates AML transaction risk based on FATF RBA, CDD thresholds, high-risk jurisdictions,
    and typological red flags.
    
    Args:
        amount: Transaction amount in given currency.
        currency: 3-letter currency code (e.g. USD, EUR).
        country: Originating jurisdiction.
        entity_type: Type of customer (e.g. Individual, Shell Corp, Trust, VASP, MSB).
        payment_method: e.g., Wire Transfer, Cash, Crypto, P2P, Bearer Cheque.
        counterparty_country: Destination jurisdiction.
        
    Returns:
        Risk evaluation dictionary containing score (0-100), risk tier, triggered red flags, and regulatory requirements.
    """
    db = load_sanctions_data()
    black_list = [c.lower() for c in db.get("high_risk_jurisdictions", {}).get("black_list", [])]
    grey_list = [c.lower() for c in db.get("high_risk_jurisdictions", {}).get("grey_list", [])]
    
    score = 10
    flags: List[str] = []
    obligations: List[str] = []
    
    # 1. Jurisdictional Risk
    c_lower = country.lower()
    cp_lower = counterparty_country.lower()
    
    if c_lower in black_list or cp_lower in black_list:
        score += 55
        flags.append(f"FATF High-Risk Jurisdiction subject to Call for Action (Black-List): {country or counterparty_country}")
        obligations.append("Apply targeted financial sanctions and strict countermeasures per FATF Recommendation 19.")
    elif c_lower in grey_list or cp_lower in grey_list:
        score += 30
        flags.append(f"FATF Jurisdictions under Increased Monitoring (Grey-List): {country or counterparty_country}")
        obligations.append("Apply Enhanced Due Diligence (EDD) proportional to country risk.")
        
    # 2. Monetary Thresholds (FATF Rec 10, Rec 16)
    if amount >= 15000:
        score += 15
        flags.append(f"Exceeds FATF Occasional Customer Due Diligence (CDD) threshold (>= $15,000 equivalent).")
        obligations.append("Full CDD identity verification and beneficial ownership inquiry required.")
    elif 9000 <= amount < 10000:
        score += 25
        flags.append("Potential structuring pattern ('smurfing'): amount clustered just below statutory $10,000 CTR reporting threshold.")
        obligations.append("Flag for suspicious transaction review and potential SAR/STR filing.")
        
    if amount >= 1000 and payment_method.lower() in ["wire transfer", "crypto", "virtual asset"]:
        obligations.append("FATF Recommendation 16 Crypto/Wire 'Travel Rule' applies: must transmit originator and beneficiary information.")

    # 3. Entity Type Risk
    entity_lower = entity_type.lower()
    if any(k in entity_lower for k in ["shell", "bearer", "offshore", "special purpose vehicle"]):
        score += 30
        flags.append(f"High-risk corporate structure: '{entity_type}' frequently utilized to obscure beneficial ownership.")
        obligations.append("Mandatory verification of natural persons with 25%+ controlling interest (Rec 24/25).")
    elif any(k in entity_lower for k in ["msb", "money service", "vasp", "crypto exchange"]):
        score += 15
        flags.append(f"Intermediary money service / VASP entity.")
        obligations.append("Perform counterparty VASP / nested correspondent banking due diligence.")

    # 4. Payment Method Risk
    pay_lower = payment_method.lower()
    if any(k in pay_lower for k in ["cash", "bearer"]):
        score += 20
        flags.append(f"Anonymous / bearer payment mechanism: {payment_method}")
    elif any(k in pay_lower for k in ["mixer", "privacy coin", "unhosted", "monero"]):
        score += 35
        flags.append(f"Anonymity-enhancing crypto mechanism detected: {payment_method}")
        obligations.append("Require cryptographic proof of wallet control or initiate SAR.")

    # Cap score at 100
    final_score = min(score, 100)
    
    if final_score >= 70:
        tier = "HIGH"
        action = "REJECT_OR_EDD: Transaction must be halted pending Enhanced Due Diligence and Compliance Officer sign-off."
    elif final_score >= 40:
        tier = "MEDIUM"
        action = "REVIEW: Transaction requires secondary analyst review and verification of source of funds."
    else:
        tier = "LOW"
        action = "AUTO_PASS: Standard automated transaction monitoring rules apply."

    return {
        "risk_score": final_score,
        "risk_tier": tier,
        "recommended_action": action,
        "triggered_flags": flags,
        "regulatory_obligations": obligations,
        "parameters": {
            "amount": amount,
            "currency": currency,
            "country": country,
            "entity_type": entity_type,
            "payment_method": payment_method,
            "counterparty_country": counterparty_country
        }
    }
