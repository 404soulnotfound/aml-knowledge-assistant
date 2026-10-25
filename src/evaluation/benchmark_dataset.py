from typing import List, Dict, Any

BENCHMARK_DATASET: List[Dict[str, Any]] = [
    # Category 1: Statutory Thresholds & Core Rules (FATF 40 Recs)
    {
        "id": "Q01",
        "category": "core_rules",
        "question": "What is the threshold ownership percentage defined by FATF for identifying a beneficial owner in a legal entity?",
        "ground_truth": "25% or more of shares, voting rights, or ownership interest.",
        "expected_source": "fatf_recommendations.md",
        "expected_citation": "Recommendation 10",
        "is_adversarial": False
    },
    {
        "id": "Q02",
        "category": "core_rules",
        "question": "What is the monetary threshold for occasional transactions requiring Customer Due Diligence (CDD) under FATF Recommendation 10?",
        "ground_truth": "USD/EUR 15,000.",
        "expected_source": "fatf_recommendations.md",
        "expected_citation": "Recommendation 10",
        "is_adversarial": False
    },
    {
        "id": "Q03",
        "category": "core_rules",
        "question": "What is the designated de minimis threshold for wire transfers and crypto transactions under the FATF Travel Rule (Recommendation 16)?",
        "ground_truth": "USD/EUR 1,000.",
        "expected_source": "fatf_recommendations.md",
        "expected_citation": "Recommendation 16",
        "is_adversarial": False
    },
    {
        "id": "Q04",
        "category": "core_rules",
        "question": "What specific senior management approval is required when establishing business relationships with foreign PEPs?",
        "ground_truth": "Senior management approval is mandatory prior to establishing or continuing a business relationship with foreign PEPs.",
        "expected_source": "fatf_recommendations.md",
        "expected_citation": "Recommendation 12",
        "is_adversarial": False
    },
    {
        "id": "Q05",
        "category": "core_rules",
        "question": "What does FATF Recommendation 20 stipulate regarding 'tipping-off' after filing a Suspicious Transaction Report?",
        "ground_truth": "Financial institutions, directors, officers, and employees are strictly prohibited by law from disclosing ('tipping-off') that an STR or related information is being reported to the FIU.",
        "expected_source": "fatf_recommendations.md",
        "expected_citation": "Recommendation 20",
        "is_adversarial": False
    },
    {
        "id": "Q06",
        "category": "core_rules",
        "question": "What are the four core Customer Due Diligence (CDD) requirements under FATF standards?",
        "ground_truth": "Identifying the customer, verifying identity using independent sources, identifying the beneficial owner (25%+ threshold), and understanding the purpose/nature of the business relationship.",
        "expected_source": "fatf_recommendations.md",
        "expected_citation": "Recommendation 10",
        "is_adversarial": False
    },
    {
        "id": "Q07",
        "category": "core_rules",
        "question": "Are financial institutions permitted to maintain anonymous bank accounts or accounts in fictitious names?",
        "ground_truth": "No, financial institutions are strictly prohibited from keeping anonymous accounts or accounts in fictitious names.",
        "expected_source": "fatf_recommendations.md",
        "expected_citation": "Recommendation 10",
        "is_adversarial": False
    },
    {
        "id": "Q08",
        "category": "core_rules",
        "question": "What measures must financial institutions take regarding the source of wealth for foreign PEPs?",
        "ground_truth": "Institutions must take reasonable measures to establish the source of wealth and source of funds.",
        "expected_source": "fatf_recommendations.md",
        "expected_citation": "Recommendation 12",
        "is_adversarial": False
    },
    {
        "id": "Q09",
        "category": "core_rules",
        "question": "What originator information must accompany cross-border wire transfers over USD/EUR 1,000?",
        "ground_truth": "Originator's name, account number or unique transaction reference, and address, national identity number, customer ID, or date/place of birth.",
        "expected_source": "fatf_recommendations.md",
        "expected_citation": "Recommendation 16",
        "is_adversarial": False
    },
    {
        "id": "Q10",
        "category": "core_rules",
        "question": "What is the primary objective of a Risk-Based Approach (RBA) under FATF Recommendation 1?",
        "ground_truth": "To identify, assess, and understand ML/TF risks, and allocate resources efficiently to mitigate higher risks with enhanced measures.",
        "expected_source": "fatf_recommendations.md",
        "expected_citation": "Recommendation 1",
        "is_adversarial": False
    },

    # Category 2: Virtual Assets & VASPs
    {
        "id": "Q11",
        "category": "vasp_crypto",
        "question": "What activities qualify an entity as a Virtual Asset Service Provider (VASP)?",
        "ground_truth": "Exchanges between virtual assets and fiat, exchanges between virtual assets, transfer of virtual assets, safekeeping/administration (custodial wallets), and financial services related to issuer sales.",
        "expected_source": "fatf_virtual_assets_guidance.md",
        "expected_citation": "Core VASP Obligations",
        "is_adversarial": False
    },
    {
        "id": "Q12",
        "category": "vasp_crypto",
        "question": "What is the 'Sunrise Issue' in the context of the FATF Crypto Travel Rule?",
        "ground_truth": "The uneven regulatory implementation timelines across jurisdictions, which does not exempt VASPs from applying due diligence when dealing with counterparties in unregulated jurisdictions.",
        "expected_source": "fatf_virtual_assets_guidance.md",
        "expected_citation": "Crypto Travel Rule",
        "is_adversarial": False
    },
    {
        "id": "Q13",
        "category": "vasp_crypto",
        "question": "What risk-based measures should VASPs apply when interacting with unhosted (non-custodial) wallets?",
        "ground_truth": "Assess counterparty risk using blockchain analytics and apply risk-based measures such as verifying wallet ownership or setting transfer limits.",
        "expected_source": "fatf_virtual_assets_guidance.md",
        "expected_citation": "Unhosted Wallets",
        "is_adversarial": False
    },
    {
        "id": "Q14",
        "category": "vasp_crypto",
        "question": "Name two key red flags associated with virtual asset transactions.",
        "ground_truth": "Structuring under $1,000 threshold, frequent transfers to mixers/privacy coins, rapid chain-hopping, or interacting with unregulated P2P platforms.",
        "expected_source": "fatf_virtual_assets_guidance.md",
        "expected_citation": "High-Risk Red Flags in Virtual Assets",
        "is_adversarial": False
    },

    # Category 3: Typologies, TBML & Correspondent Banking
    {
        "id": "Q15",
        "category": "typologies",
        "question": "Explain the four primary techniques of Trade-Based Money Laundering (TBML).",
        "ground_truth": "Over-invoicing/under-invoicing of goods, multiple invoicing of goods, short or over-shipping of goods, and phantom shipments.",
        "expected_source": "fincen_red_flags_and_typologies.md",
        "expected_citation": "Trade-Based Money Laundering",
        "is_adversarial": False
    },
    {
        "id": "Q16",
        "category": "typologies",
        "question": "What is a 'Funnel Account' and how does it facilitate structuring?",
        "ground_truth": "An account that receives numerous cash deposits below threshold from multiple people at one branch, with immediate cash withdrawals or transfers at a distant branch.",
        "expected_source": "fincen_red_flags_and_typologies.md",
        "expected_citation": "Funnel Accounts",
        "is_adversarial": False
    },
    {
        "id": "Q17",
        "category": "typologies",
        "question": "What are the Wolfsberg principles regarding business relationships with Shell Banks?",
        "ground_truth": "Financial institutions must strictly refuse to enter into or maintain correspondent banking relationships with shell banks.",
        "expected_source": "wolfsberg_cdd_principles.md",
        "expected_citation": "Shell Banks",
        "is_adversarial": False
    },
    {
        "id": "Q18",
        "category": "typologies",
        "question": "What review frequency does the Wolfsberg Group recommend for High-Risk customers versus Low-Risk customers?",
        "ground_truth": "High risk requires annual or event-driven reviews, while low risk requires reviews every 3 to 5 years.",
        "expected_source": "wolfsberg_cdd_principles.md",
        "expected_citation": "Periodic CDD Reviews",
        "is_adversarial": False
    },

    # Category 4: Adversarial / Unanswerable / Hallucination-Inducing Questions
    {
        "id": "Q19",
        "category": "adversarial",
        "question": "Under FATF Recommendation 52, what is the mandatory penalty fee in Bitcoin for failure to report smart contracts?",
        "ground_truth": "FATF has only 40 recommendations; Recommendation 52 does not exist, and there is no mandatory fee in Bitcoin.",
        "expected_source": "None",
        "expected_citation": "None",
        "is_adversarial": True
    },
    {
        "id": "Q20",
        "category": "adversarial",
        "question": "What are the FATF guidelines for extraterrestrial bank branches operating on Mars colonies?",
        "ground_truth": "The provided AML guidance documents do not contain information regarding space or Mars; no statutory guidance exists.",
        "expected_source": "None",
        "expected_citation": "None",
        "is_adversarial": True
    },
    {
        "id": "Q21",
        "category": "adversarial",
        "question": "Does FATF Recommendation 10 state that domestic retail customers are exempt from identity verification if they deposit under $50,000?",
        "ground_truth": "No, that is false. The designated threshold is USD/EUR 15,000 for occasional transactions and CDD is always required when establishing business relations.",
        "expected_source": "fatf_recommendations.md",
        "expected_citation": "Recommendation 10",
        "is_adversarial": True
    },
    {
        "id": "Q22",
        "category": "adversarial",
        "question": "What is the FinCEN regulation regarding the use of teleportation devices to move physical cash across international borders?",
        "ground_truth": "The provided AML documents contain no information regarding teleportation devices.",
        "expected_source": "None",
        "expected_citation": "None",
        "is_adversarial": True
    },
    {
        "id": "Q23",
        "category": "adversarial",
        "question": "According to the Wolfsberg Group, are banks allowed to disclose the filing of a suspicious activity report if the customer promises to pay a compliance penalty?",
        "ground_truth": "No, tipping-off is strictly prohibited by law under FATF Recommendation 20 and Wolfsberg standards under any circumstances.",
        "expected_source": "fatf_recommendations.md",
        "expected_citation": "Recommendation 20",
        "is_adversarial": True
    },
    {
        "id": "Q24",
        "category": "adversarial",
        "question": "What is the statutory FATF threshold for anonymous prepaid cards on the Intergalactic Monetary Network?",
        "ground_truth": "The provided guidance does not cover an Intergalactic Monetary Network.",
        "expected_source": "None",
        "expected_citation": "None",
        "is_adversarial": True
    },
    {
        "id": "Q25",
        "category": "adversarial",
        "question": "Under FATF Recommendation 75, how many days does an FIU have to seize sovereign gold certificates?",
        "ground_truth": "FATF only has 40 Recommendations; Recommendation 75 is fictitious.",
        "expected_source": "None",
        "expected_citation": "None",
        "is_adversarial": True
    }
]
