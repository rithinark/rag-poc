"""
Workplace & Employee Policies Dataset.
Curated from open enterprise benchmarks (including the open-source 37signals / Basecamp handbook).
Provides accessible, intuitive documentation covering benefits, remote work, equipment,
parental leave, summer hours, and career progression.
"""

from __future__ import annotations

from typing import List
from rag_poc.core.models import ClassificationLevel, Document


def get_workplace_documents(tenant_id: str = "tenant_alpha") -> List[Document]:
    """
    Returns foundational employee handbook documents for workplace policies,
    benefits, remote equipment stipends, and leave entitlements.
    """
    return [
        Document(
            document_id=f"{tenant_id}_doc_hr_benefits_01",
            title="Employee Health, Wellness & Retirement Benefits",
            source_uri="doc://hr/benefits-and-healthcare.md",
            tenant_id=tenant_id,
            security_groups=["default", "security_team", "hr_team"],
            classification=ClassificationLevel.INTERNAL,
            metadata={"domain": "hr_benefits", "version": "2026.1"},
            content=(
                "# Employee Health, Wellness & Retirement Benefits\n\n"
                "## Medical Insurance\n"
                "Medical insurance in the United States is provided through Blue Cross Blue Shield PPO. "
                "The company pays 75% of the total premium, and the employee contributes the remaining 25% via payroll deduction. "
                "Coverage begins on your first calendar day of employment. Open enrollment takes place each November, "
                "with updated elections taking effect on December 1. Marriages, domestic partnerships, and civil unions are equally eligible.\n\n"
                "Bi-weekly payroll deduction schedule:\n"
                "- Employee-only coverage: $96.05 per pay period\n"
                "- Employee + Partner coverage: $197.27 per pay period\n"
                "- Employee + Children coverage: $186.44 per pay period\n"
                "- Full Family coverage: $287.65 per pay period\n\n"
                "Staff outside the United States paying out of pocket for private health insurance receive monthly reimbursement "
                "up to 75% of their premium, capped at the US group plan benchmark.\n\n"
                "## Dental & Vision Coverage\n"
                "Comprehensive dental and vision insurance is administered via MetLife. "
                "The company covers 100% of the premium for both employees and eligible dependents. "
                "Dental covers bi-annual preventive cleanings, annual X-rays, and up to $2,500 in annual restorative procedures. "
                "Vision covers an annual comprehensive exam and up to $250 for prescription eyewear.\n\n"
                "## 401(k) Retirement Savings & Company Match\n"
                "The corporate retirement plan is administered through Vanguard. "
                "The company provides a 100% dollar-for-dollar match on employee contributions up to 6% of your base salary. "
                "Employees are eligible to enroll starting with their first paycheck, and company matching vests immediately.\n\n"
                "## Life & Disability Insurance\n"
                "A $400,000 term life and AD&D policy is provided through MetLife at no cost to the employee (100% company-paid). "
                "Short-Term Disability (STD) replaces up to 70% of pre-disability earnings up to $3,000 per week for up to 12 weeks "
                "following a 7-day waiting period. Long-Term Disability (LTD) replaces up to 60% of earnings up to $12,000 per month "
                "following a 90-day waiting period."
            ),
        ),
        Document(
            document_id=f"{tenant_id}_doc_workplace_remote_01",
            title="Remote Work Setup, Office Stipends & Hardware Policy",
            source_uri="doc://workplace/remote-work-and-stipends.md",
            tenant_id=tenant_id,
            security_groups=["default", "security_team", "hr_team"],
            classification=ClassificationLevel.INTERNAL,
            metadata={"domain": "workplace_equipment", "version": "2026.1"},
            content=(
                "# Remote Work Setup, Office Stipends & Hardware Policy\n\n"
                "## Work from Anywhere Philosophy\n"
                "Our workforce is 100% distributed. Employees may work from any location with reliable high-speed internet. "
                "When moving across state, provincial, or international borders, employees must notify People Operations at least "
                "30 days prior to relocation to ensure payroll tax, residency, and labor law compliance.\n\n"
                "## Home Office Setup Stipend\n"
                "Every new employee receives an initial home office setup budget of up to $3,000 during their first 12 months. "
                "This allowance covers:\n"
                "- Ergonomic desk or motorized sit-stand converter\n"
                "- Ergonomic task chair (e.g., Herman Miller, Steelcase)\n"
                "- External 4K/5K displays, monitor arms, and webcam\n"
                "- Keyboard, mouse, desk lighting, and noise-canceling headphones\n\n"
                "## Coworking Space Allowance\n"
                "Employees who prefer working outside the home may expense up to $200 per month for a membership at a local coworking space. "
                "Alternatively, employees can request a company-provided Deskpass subscription for flexible day desk reservations.\n\n"
                "## Computer Hardware & IT Provisioning\n"
                "Every technical and operational team member receives a brand-new Apple MacBook Pro upon onboarding. "
                "Laptops are enrolled in Kandji Mobile Device Management (MDM) with FileVault encryption enabled. "
                "Laptops are on an automatic 3-year hardware refresh cycle.\n\n"
                "## Corporate Expense Card (Airbase)\n"
                "Staff members are issued a corporate Visa card via Airbase for work-related expenses, software tools, reference books, "
                "and company travel. Receipts must be uploaded into Airbase for any single transaction exceeding $75."
            ),
        ),
        Document(
            document_id=f"{tenant_id}_doc_hr_leave_01",
            title="Paid Time Off, Summer Hours & Parental Leave",
            source_uri="doc://hr/leave-and-time-off.md",
            tenant_id=tenant_id,
            security_groups=["default", "security_team", "hr_team"],
            classification=ClassificationLevel.INTERNAL,
            metadata={"domain": "hr_leave", "version": "2026.1"},
            content=(
                "# Paid Time Off, Summer Hours & Parental Leave\n\n"
                "## Paid Time Off (PTO) & Holidays\n"
                "All full-time team members receive 4 weeks (20 business days) of paid vacation annually, "
                "in addition to recognized national federal holidays. PTO is accrued pro-rata and front-loaded on January 1.\n\n"
                "## Summer Hours (4-Day Work Week)\n"
                "From May 1 through August 31 each year, the company observes summer hours: a 4-day, 32-hour work week. "
                "Employees take Fridays off with 100% full regular pay. Teams coordinate coverage for critical production incidents "
                "via rotating on-call shifts, which receive compensatory time off.\n\n"
                "## Fully Paid Parental Leave\n"
                "New parents (birthing parent, non-birthing parent, adoptive, or foster placement) are entitled to 16 weeks of 100% "
                "paid parental leave. Employees become eligible after completing 6 months of continuous employment. "
                "Leave may be taken consecutively or split across the first 12 months following birth or adoption.\n\n"
                "## Sick & Mental Health Days\n"
                "We provide flexible paid sick leave for personal illness, medical appointments, and mental wellness rest. "
                "No doctor's note or medical certificate is required for absences of up to 3 consecutive business days.\n\n"
                "## Paid Sabbatical Policy\n"
                "For every 3 years of continuous service, full-time employees are awarded a 30-day (1 full month) paid sabbatical "
                "to unplug, travel, pursue creative projects, or spend time with family. The company also contributes a $2,000 travel bonus."
            ),
        ),
        Document(
            document_id=f"{tenant_id}_doc_people_career_01",
            title="Career Progression, Review Cycles & Mastery Levels",
            source_uri="doc://people/career-growth-and-leveling.md",
            tenant_id=tenant_id,
            security_groups=["default", "security_team", "hr_team"],
            classification=ClassificationLevel.INTERNAL,
            metadata={"domain": "people_career", "version": "2026.1"},
            content=(
                "# Career Progression, Review Cycles & Mastery Levels\n\n"
                "## First-Year Performance Review Cadence\n"
                "During your first year, you will participate in three structured performance check-ins with your manager:\n"
                "1. **3-Month Review**: Alignment on initial onboarding, early project deliverables, and team dynamics.\n"
                "2. **6-Month Review**: Mid-year evaluation assessing execution velocity, problem-solving, and independence.\n"
                "3. **12-Month Review**: Comprehensive first-anniversary review confirming long-term career growth trajectory.\n\n"
                "Following the first anniversary, formal reviews transition to an annual review schedule held on your hiring anniversary date.\n\n"
                "## The Five Mastery Levels\n"
                "Career ladders across engineering, design, and operations are structured around five distinct levels of mastery:\n"
                "- **Level 1 (Junior)**: Proficient in foundational tools and workflows; handles routine, well-scoped tasks with guidance.\n"
                "- **Level 2 (Mid-Level)**: Confidently delivers scoped features independently; writes high-quality, maintainable code.\n"
                "- **Level 3 (Senior)**: Takes complex features from concept to production delivery as sole owner; cross-team mentor.\n"
                "- **Level 4 (Lead)**: Subject matter expert; serves as technical lead; reviews critical architectural designs and PRs.\n"
                "- **Level 5 (Principal)**: Sets strategic direction across whole disciplines; invents novel architectures; reports directly to leadership.\n\n"
                "Advancement is based on demonstrated impact, craftsmanship, and peer respect rather than tenure or managing direct reports."
            ),
        ),
        Document(
            document_id=f"{tenant_id}_doc_it_security_01",
            title="IT Helpdesk, Authentication & Device Security Hygiene",
            source_uri="doc://it/helpdesk-and-security.md",
            tenant_id=tenant_id,
            security_groups=["default", "security_team", "hr_team"],
            classification=ClassificationLevel.INTERNAL,
            metadata={"domain": "it_helpdesk", "version": "2026.1"},
            content=(
                "# IT Helpdesk, Authentication & Device Security Hygiene\n\n"
                "## Password & Vault Management\n"
                "All employees are provisioned an enterprise 1Password account. "
                "Master passwords must be a minimum of 16 characters (or a 4-word random passphrase). "
                "Passwords must never be reused across personal and corporate services.\n\n"
                "## Multi-Factor Authentication (MFA) Standards\n"
                "MFA is mandatory across all internal corporate tools (Google Workspace, GitHub, AWS, Slack). "
                "Approved MFA methods are FIDO2/WebAuthn hardware security keys (e.g., YubiKey) and time-based one-time password (TOTP) "
                "authenticator apps. SMS/voice-based 2FA is prohibited due to SIM-swapping risks.\n\n"
                "## Device Security & Screen Lock Requirements\n"
                "All company-managed workstations must enforce:\n"
                "- FileVault full-disk encryption with recovery keys escrowed in Kandji MDM.\n"
                "- Automated screen lock after 5 minutes of inactivity.\n"
                "- Operating system and browser security updates applied within 7 days of public release.\n\n"
                "## Lost or Stolen Equipment Escalation\n"
                "If a company-issued laptop, tablet, or phone is lost or stolen, the employee must contact IT Helpdesk and Security "
                "within 1 hour via email at `security@enterprise.internal` or telephone hotline +1-800-555-0199. "
                "IT Security will initiate an immediate remote cryptographic wipe."
            ),
        ),
    ]
