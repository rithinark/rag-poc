"""
Hugging Face E-Commerce & Customer Support Dataset Ingestion.
Dataset: bitext/Bitext-customer-support-llm-chatbot-training-dataset
License: CDLA-Sharing-1.0
Domain: Customer Support, Retail Orders, Refunds, Payments & Shipping
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from rag_poc.core.models import ClassificationLevel, Document


DATASET_FILE = Path(__file__).resolve().parents[3] / "data" / "hf_datasets" / "bitext_customer_support" / "dataset_sample.jsonl"


def get_customer_support_documents(tenant_id: str = "tenant_alpha") -> List[Document]:
    """
    Returns structured enterprise customer support knowledge documents derived from
    the Hugging Face bitext customer support benchmark dataset.
    """
    return [
        Document(
            document_id=f"{tenant_id}_doc_retail_orders_01",
            title="Online Orders, Modifications, Cancellations & Order Tracking",
            source_uri="doc://ecommerce/orders-and-cancellations.md",
            tenant_id=tenant_id,
            security_groups=["default", "support_agents", "customers"],
            classification=ClassificationLevel.INTERNAL,
            metadata={
                "source": "huggingface:bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                "domain": "customer_support_orders",
                "version": "2026.1",
            },
            content=(
                "# Online Orders, Modifications, Cancellations & Order Tracking\n\n"
                "## How to Cancel an Order\n"
                "Customers can cancel an active order directly through the online self-service customer portal. "
                "To initiate a cancellation:\n"
                "1. Log into your account and navigate to 'My Orders' or 'Order History'.\n"
                "2. Locate the specific order by order number.\n"
                "3. Click the 'Cancel Order' button under the actions menu.\n"
                "4. Select the reason for cancellation and confirm your request.\n"
                "A cancellation confirmation email will be delivered to your registered email address within 5 minutes.\n\n"
                "## Cancellation Cut-off & Fees\n"
                "Orders can be cancelled with a 100% full refund and $0 cancellation fee prior to warehouse dispatch. "
                "Once an order enters the packing and palletization phase (typically 60 minutes after order placement), "
                "cancellations cannot be guaranteed. If the item has already shipped, customers must wait for delivery "
                "and initiate a standard 30-day return. Orders cancelled after freight dispatch may be subject to a 15% restocking fee.\n\n"
                "## Modifying an Order\n"
                "Item quantities, product sizes, or color selections can be edited within 60 minutes of placing the order. "
                "After 60 minutes, orders are routed to the fulfillment queue and cannot be altered.\n\n"
                "## Tracking Your Shipment\n"
                "Real-time tracking is provided for every package. Upon dispatch, a tracking link from our carrier partners "
                "(FedEx, UPS, or DHL) will be sent via email and SMS. You can also view live tracking coordinates and estimated "
                "time of arrival (ETA) directly from the Order Details dashboard."
            ),
        ),
        Document(
            document_id=f"{tenant_id}_doc_retail_refunds_01",
            title="Return Policy, Money-Back Guarantee & Refund Processing",
            source_uri="doc://ecommerce/refunds-and-returns.md",
            tenant_id=tenant_id,
            security_groups=["default", "support_agents", "customers"],
            classification=ClassificationLevel.INTERNAL,
            metadata={
                "source": "huggingface:bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                "domain": "customer_support_refunds",
                "version": "2026.1",
            },
            content=(
                "# Return Policy, Money-Back Guarantee & Refund Processing\n\n"
                "## 30-Day Money-Back Guarantee\n"
                "We provide a 30-day return window on all purchases. Eligible items must be in new, unused condition "
                "with original tags, accessories, and retail packaging intact.\n\n"
                "## Non-Refundable Items\n"
                "The following items cannot be returned or refunded:\n"
                "- Digital gift cards and prepaid vouchers\n"
                "- Custom-engraved or personalized merchandise\n"
                "- Clearance items designated as 'Final Sale'\n"
                "- Opened software licenses and digital downloads\n\n"
                "## Refund Processing Timelines\n"
                "Once returned merchandise is received at our return center, our intake team inspects the items within "
                "2 business days. Approved refunds are processed immediately and credited back to the original method of payment. "
                "Depending on your financial institution, funds appear on your bank statement within:\n"
                "- Credit and Debit Cards: 3 to 5 business days\n"
                "- PayPal / Apple Pay: 24 to 48 hours\n"
                "- Store Credit & Gift Cards: Instant (immediate wallet reload)\n\n"
                "## Damaged, Defective or Incorrect Items\n"
                "If you receive a defective or damaged product, notify customer support within 48 hours of delivery. "
                "We provide a prepaid return shipping label and issue an immediate zero-cost replacement or 100% full refund "
                "including original freight charges."
            ),
        ),
        Document(
            document_id=f"{tenant_id}_doc_retail_payments_01",
            title="Accepted Payment Methods, Billing & Troubleshooting Payment Errors",
            source_uri="doc://ecommerce/payments-and-billing.md",
            tenant_id=tenant_id,
            security_groups=["default", "support_agents", "finance"],
            classification=ClassificationLevel.INTERNAL,
            metadata={
                "source": "huggingface:bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                "domain": "customer_support_payments",
                "version": "2026.1",
            },
            content=(
                "# Accepted Payment Methods, Billing & Troubleshooting Payment Errors\n\n"
                "## Supported Payment Methods\n"
                "Our checkout system securely supports the following payment options:\n"
                "- Credit & Debit Cards: Visa, MasterCard, American Express, Discover, JCB\n"
                "- Digital Wallets: Apple Pay, Google Pay, PayPal, Shop Pay\n"
                "- Installment Financing: Klarna (4 interest-free installments)\n"
                "- Official Store Gift Cards and Store Credit balances\n\n"
                "## Resolving Payment Errors & Card Declines\n"
                "If your payment is declined during checkout, please check the following common issues:\n"
                "1. Billing Address Match: Ensure the postal/ZIP code entered matches the billing address on file with your bank.\n"
                "2. CVV / Security Code: Verify the 3-digit CVV on the back of Visa/MasterCard, or 4-digit code on the front of Amex.\n"
                "3. International Restrictions: Foreign transactions may trigger automated bank security filters; contact your bank to approve the transaction.\n"
                "4. 3D Secure Authentication: Complete the SMS OTP or biometric verification prompt requested by your card issuer.\n"
                "If difficulties persist, try using PayPal or Apple Pay, or contact our 24/7 billing support line at 1-800-555-0199.\n\n"
                "## Invoices & Receipts\n"
                "Itemized PDF receipts including tax breakdown (VAT / GST / State Sales Tax) and company tax identification "
                "numbers are automatically generated upon payment settlement. You can download invoices anytime under Account Settings > Invoices."
            ),
        ),
        Document(
            document_id=f"{tenant_id}_doc_retail_shipping_01",
            title="Shipping Options, Delivery Timelines & Address Updates",
            source_uri="doc://ecommerce/shipping-and-delivery.md",
            tenant_id=tenant_id,
            security_groups=["default", "support_agents", "logistics"],
            classification=ClassificationLevel.INTERNAL,
            metadata={
                "source": "huggingface:bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                "domain": "customer_support_shipping",
                "version": "2026.1",
            },
            content=(
                "# Shipping Options, Delivery Timelines & Address Updates\n\n"
                "## Delivery Tiers & Shipping Rates\n"
                "We provide multiple shipping tiers across the continental United States and internationally:\n"
                "- Standard Delivery (3-5 business days): Free for orders over $50; flat rate of $4.99 for orders under $50.\n"
                "- Express 2-Day Shipping (2 business days): Flat rate of $12.99.\n"
                "- Priority Overnight (Next business day by 10:30 AM): Flat rate of $24.99.\n"
                "- International Standard (7-14 business days): Calculated dynamically at checkout with duties prepaid (DDP).\n\n"
                "## Daily Order Cut-off Times\n"
                "Orders placed Monday through Friday prior to 2:00 PM EST ship the same day. Orders placed after 2:00 PM EST, "
                "or on weekends and federal holidays, ship the next business day.\n\n"
                "## Changing Delivery Address\n"
                "Customers can update their shipping destination address within 60 minutes of placing an order by navigating "
                "to Order History > Edit Shipping Address. Once the parcel is in transit, the delivery address cannot be changed "
                "directly on our site; please use FedEx Delivery Manager or UPS My Choice to reroute or hold the package at a nearby pickup locker."
            ),
        ),
        Document(
            document_id=f"{tenant_id}_doc_retail_support_01",
            title="Customer Helpdesk Hours, Live Chat & Account Registration",
            source_uri="doc://ecommerce/helpdesk-and-account.md",
            tenant_id=tenant_id,
            security_groups=["default", "support_agents", "customers"],
            classification=ClassificationLevel.INTERNAL,
            metadata={
                "source": "huggingface:bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                "domain": "customer_support_helpdesk",
                "version": "2026.1",
            },
            content=(
                "# Customer Helpdesk Hours, Live Chat & Account Registration\n\n"
                "## Contact Channels & Support Availability\n"
                "Our dedicated customer service team is accessible through multiple support channels:\n"
                "- Live Chat Support: 24 hours a day, 7 days a week directly on our web store and mobile app.\n"
                "- Telephone Support: Toll-free 1-800-555-0199, available Monday through Friday from 8:00 AM to 8:00 PM EST.\n"
                "- Email Support: support@retailshop.com with guaranteed response within 4 hours during business days.\n\n"
                "## Account Registration & Password Recovery\n"
                "Creating an account is free and enables 1-click checkout, saved addresses, and order history tracking. "
                "If you experience registration problems or forget your password:\n"
                "1. Click 'Forgot Password' on the login portal.\n"
                "2. Enter your account email address to receive a secure, one-time password reset token.\n"
                "3. Reset tokens expire after 15 minutes for cybersecurity protection.\n"
                "If your account is locked due to repeated incorrect password attempts, contact our 24/7 live chat team for identity verification and account unlock."
            ),
        ),
    ]


def load_raw_hf_samples() -> List[dict]:
    """
    Reads the raw Hugging Face JSONL sample records from disk.
    """
    if not DATASET_FILE.exists():
        return []
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]
