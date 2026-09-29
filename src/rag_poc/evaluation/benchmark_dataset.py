"""
Curated Evaluation Benchmark Dataset for Enterprise RAGAS Evaluation.
Contains verified ground truth answers across all loaded knowledge domains:
- Workplace & HR Policies
- E-Commerce & Customer Support (Hugging Face)
- Security & Key Management
"""

from __future__ import annotations

from typing import Any, Dict, List

BENCHMARK_CASES: List[Dict[str, Any]] = [
    # --- Workplace & HR Policies ---
    {
        "id": "hr_01",
        "domain": "Workplace & HR Policies",
        "tenant_id": "tenant_alpha",
        "user_id": "employee@enterprise.internal",
        "security_groups": ["default", "hr_team"],
        "query": "What is the 401(k) company match policy and medical insurance coverage?",
        "ground_truth": (
            "The company provides a 100% dollar-for-dollar match on employee contributions up to 6% of base salary via Vanguard, "
            "vesting immediately starting with your first paycheck. Medical insurance is provided through Blue Cross Blue Shield PPO, "
            "where the company pays 75% of the total premium and the employee pays 25% via payroll deduction."
        ),
        "key_facts": ["100%", "6%", "Vanguard", "75%", "25%", "Blue Cross Blue Shield"],
    },
    {
        "id": "hr_02",
        "domain": "Workplace & HR Policies",
        "tenant_id": "tenant_alpha",
        "user_id": "employee@enterprise.internal",
        "security_groups": ["default", "hr_team"],
        "query": "What is the first-year home office setup stipend and coworking space allowance?",
        "ground_truth": (
            "Full-time remote employees receive a $3,000 first-year home office setup stipend, an annual $500 refresh allowance, "
            "and a monthly $200 stipend for local coworking space memberships charged to an Airbase Visa commercial card."
        ),
        "key_facts": ["$3,000", "$500", "$200", "Airbase"],
    },
    {
        "id": "hr_03",
        "domain": "Workplace & HR Policies",
        "tenant_id": "tenant_alpha",
        "user_id": "employee@enterprise.internal",
        "security_groups": ["default", "hr_team"],
        "query": "What is the company policy on summer hours and paid parental leave?",
        "ground_truth": (
            "From May 1 through August 31, the company operates on a 4-day, 32-hour work week with every Friday off at full salary. "
            "The company provides 16 weeks of 100% paid parental leave for birthing parents, non-birthing parents, and adoptive parents."
        ),
        "key_facts": ["Summer Hours", "4-day", "32-hour", "Fridays off", "16 weeks", "100% paid"],
    },

    # --- Customer Support & E-Commerce (Hugging Face) ---
    {
        "id": "cs_01",
        "domain": "Customer Support (Hugging Face)",
        "tenant_id": "tenant_alpha",
        "user_id": "support-agent@retailshop.internal",
        "security_groups": ["support_agents", "default", "customers"],
        "query": "How do I cancel an order and what is the cancellation cutoff time?",
        "ground_truth": (
            "Customers can cancel an active order directly from 'My Orders' in the customer portal. "
            "Cancellations are free with a 100% full refund if completed within 60 minutes of placing the order before warehouse packaging. "
            "Orders cancelled after freight dispatch may incur a 15% restocking fee."
        ),
        "key_facts": ["My Orders", "60 minutes", "100% full refund", "$0 cancellation fee", "15% restocking fee"],
    },
    {
        "id": "cs_02",
        "domain": "Customer Support (Hugging Face)",
        "tenant_id": "tenant_alpha",
        "user_id": "support-agent@retailshop.internal",
        "security_groups": ["support_agents", "default", "customers"],
        "query": "What is the 30-day refund policy and how long does it take for funds to appear on my statement?",
        "ground_truth": (
            "We offer a 30-day money-back guarantee for unused items in original packaging. Once our return center inspects the items "
            "(within 2 business days), refunds are credited to credit/debit cards within 3 to 5 business days, and 24 to 48 hours for PayPal."
        ),
        "key_facts": ["30-day", "2 business days", "3 to 5 business days", "PayPal", "24 to 48 hours"],
    },
    {
        "id": "cs_03",
        "domain": "Customer Support (Hugging Face)",
        "tenant_id": "tenant_alpha",
        "user_id": "support-agent@retailshop.internal",
        "security_groups": ["support_agents", "default", "customers"],
        "query": "What payment methods are accepted and how do I troubleshoot a card decline?",
        "ground_truth": (
            "Supported methods include Visa, MasterCard, American Express, Discover, PayPal, Apple Pay, Google Pay, and Klarna. "
            "For card declines, verify your billing ZIP code matches your bank record, confirm the 3-digit CVV, check for international restrictions, "
            "or complete the 3D Secure SMS OTP prompt."
        ),
        "key_facts": ["Visa", "MasterCard", "PayPal", "Apple Pay", "billing ZIP code", "CVV", "3D Secure"],
    },
    {
        "id": "cs_04",
        "domain": "Customer Support (Hugging Face)",
        "tenant_id": "tenant_alpha",
        "user_id": "support-agent@retailshop.internal",
        "security_groups": ["support_agents", "default", "customers"],
        "query": "What shipping speeds and delivery tiers are available, and how can I update my delivery address?",
        "ground_truth": (
            "Standard shipping takes 3-5 business days (free over $50, otherwise $4.99). Express 2-Day is $12.99 and Priority Overnight is $24.99. "
            "Orders before 2:00 PM EST ship same business day. Delivery addresses can be changed in Order History within 60 minutes of placing the order."
        ),
        "key_facts": ["Standard", "3-5 business days", "free over $50", "2:00 PM EST", "60 minutes"],
    },

    # --- Security & Key Management ---
    {
        "id": "sec_01",
        "domain": "Security & Key Management",
        "tenant_id": "tenant_alpha",
        "user_id": "security-analyst@enterprise.internal",
        "security_groups": ["security_team", "default"],
        "query": "What is the AES-256 key rotation policy for payment transactions?",
        "ground_truth": (
            "All AES-256 cryptographic master keys and encryption keys must be rotated every 90 days in compliance with enterprise security policy. "
            "Data Encryption Keys (DEKs) are rotated every 30 days in memory, and Key Encryption Keys (KEKs) are rotated every 90 days in HSM."
        ),
        "key_facts": ["AES-256", "90 days", "DEK", "30 days", "KEK", "HSM"],
    },
    {
        "id": "sec_02",
        "domain": "Security & Key Management",
        "tenant_id": "tenant_alpha",
        "user_id": "security-analyst@enterprise.internal",
        "security_groups": ["security_team", "default"],
        "query": "How are KMS cryptographic backup master keys stored?",
        "ground_truth": (
            "KMS cryptographic backup master keys are stored in an encrypted offline air-gapped vault. "
            "Access requires multi-party quorum authorization (M-of-N Shamir secret sharing requiring 3 of 5 executive key holders)."
        ),
        "key_facts": ["encrypted offline air-gapped vault", "Shamir secret sharing", "3 of 5"],
    },
]
