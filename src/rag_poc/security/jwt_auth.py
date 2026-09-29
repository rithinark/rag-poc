"""
JWT Authentication & Caller Identity Extraction.
Validates OIDC/OAuth2 tokens and extracts cryptographically signed tenant and security claims.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import jwt

from rag_poc.core.config import settings
from rag_poc.core.models import ClassificationLevel, JWTClaims


class AuthenticationError(Exception):
    """Raised when JWT verification fails or token is invalid."""
    pass


class JWTAuthenticator:
    """
    Validates cryptographically signed JSON Web Tokens (JWT) and extracts
    caller identity, multi-tenant boundaries, and security group clearance.
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: Optional[str] = None,
        issuer: Optional[str] = None,
        audience: Optional[str] = None,
    ):
        self.secret_key = secret_key or settings.jwt_secret_key
        self.algorithm = algorithm or settings.jwt_algorithm
        self.issuer = issuer or settings.jwt_issuer
        self.audience = audience or settings.jwt_audience

    def generate_token(
        self,
        sub: str,
        tenant_id: str,
        roles: Optional[List[str]] = None,
        security_groups: Optional[List[str]] = None,
        clearance_level: ClassificationLevel = ClassificationLevel.INTERNAL,
        expires_in_seconds: int = 3600,
    ) -> str:
        """Utility for creating signed test tokens."""
        now = int(time.time())
        payload = {
            "sub": sub,
            "tenant_id": tenant_id,
            "roles": roles or ["user"],
            "security_groups": security_groups or ["default"],
            "clearance_level": clearance_level.value,
            "iss": self.issuer,
            "aud": self.audience,
            "iat": now,
            "exp": now + expires_in_seconds,
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> JWTClaims:
        """
        Validates token signature, expiration, issuer, audience, and extracts caller claims.
        """
        try:
            payload: Dict[str, Any] = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                issuer=self.issuer,
                audience=self.audience,
            )
        except jwt.ExpiredSignatureError as e:
            raise AuthenticationError("Caller JWT token has expired") from e
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid JWT token signature or format: {str(e)}") from e

        try:
            return JWTClaims(
                sub=payload["sub"],
                tenant_id=payload["tenant_id"],
                roles=payload.get("roles", []),
                security_groups=payload.get("security_groups", ["default"]),
                clearance_level=ClassificationLevel(payload.get("clearance_level", "INTERNAL")),
            )
        except Exception as e:
            raise AuthenticationError(f"JWT payload does not conform to security schema: {str(e)}") from e
