"""
Security Broker Pattern.
Enforces pre-retrieval metadata scoping by compiling verified JWT claims into
vector database boolean expression filters, preventing Context Starvation and cross-tenant leakage.
"""

from __future__ import annotations

import re
from typing import List, Optional

from rag_poc.core.models import ClassificationLevel, JWTClaims, RetrievalFilter
from rag_poc.security.jwt_auth import AuthenticationError, JWTAuthenticator


class SecurityBroker:
    """
    Translates authenticated caller identity into mathematical search boundaries
    prior to vector database index traversal.
    """

    SAFE_IDENTIFIER = re.compile(r"^[a-zA-Z0-9_-]+$")

    def __init__(self, authenticator: Optional[JWTAuthenticator] = None):
        self.authenticator = authenticator or JWTAuthenticator()

    def authenticate_and_build_filter(self, token: str) -> RetrievalFilter:
        """
        Validates the JWT token and constructs a strict RetrievalFilter.
        """
        claims = self.authenticator.verify_token(token)
        return self.claims_to_filter(claims)

    def claims_to_filter(self, claims: JWTClaims) -> RetrievalFilter:
        """
        Converts validated JWT claims into a structured RetrievalFilter object.
        """
        # Validate that tenant_id and groups contain only safe characters (prevent expression injection)
        if not self.SAFE_IDENTIFIER.match(claims.tenant_id):
            raise AuthenticationError(f"Potentially malicious characters in tenant_id: {claims.tenant_id}")

        safe_groups: List[str] = []
        for g in claims.security_groups:
            if self.SAFE_IDENTIFIER.match(g):
                safe_groups.append(g)

        return RetrievalFilter(
            tenant_id=claims.tenant_id,
            authorized_groups=safe_groups or ["default"],
            clearance_level=claims.clearance_level,
        )

    def compile_milvus_expression(self, filter_obj: RetrievalFilter) -> str:
        """
        Compiles the RetrievalFilter into a native Milvus boolean filter string.
        Evaluated during cuVS GPU / CPU index traversal.
        """
        # 1. Exact tenant boundary
        tenant_expr = f'tenant_id == "{filter_obj.tenant_id}"'

        # 2. Permitted classifications based on caller clearance
        allowed_classes = [
            f'"{lvl.value}"'
            for lvl in ClassificationLevel
            if filter_obj.clearance_level.allows(lvl)
        ]
        class_expr = f'classification in [{", ".join(allowed_classes)}]'

        # 3. Security groups: document must have at least one group matching caller groups
        groups_list_str = ", ".join(f'"{g}"' for g in filter_obj.authorized_groups)
        # Note: In Milvus with Array/JSON or string matching, JSON_CONTAINS or equality
        groups_expr = f'security_group in [{groups_list_str}]'

        return f"({tenant_expr}) and ({class_expr}) and ({groups_expr})"
