# Core Development Practices & Architecture Rules

All engineering and code contributions in this repository must strictly adhere to the following non-negotiable development practices:

---

## Rule 1: Never Patch Actual Code to Support Unit Test Cases
- **No Test-Specific Shims in Production**: Never introduce mock flags, `if is_test:` conditionals, artificial bypass branches, or watered-down security logic into production modules simply to make unit tests pass.
- **Test Real Implementations**: Production code must be written for production correctness. Unit and integration tests must validate real behavior through explicit dependency injection, polymorphic abstractions (e.g., standard interfaces and drivers), or clean test fixtures.
- **Fail Early**: If a test cannot run because of an architectural constraint, improve the architecture or design test doubles using standard dependency injection—do not contaminate production code paths.

---

## Rule 2: Never Add Try/Catch Blocks for Imports
- **No `try...except ImportError` Fallbacks**: Never hide missing dependencies behind `try: import foo except ImportError:` blocks or silent fallback stubs in production modules.
- **Explicit Dependencies**: If a dependency or library is required for a component (e.g., `presidio-analyzer`, `pymilvus`, `fastmcp`), declare it explicitly in `pyproject.toml` (or `requirements.txt`), install it in the development environment, and write direct top-level imports.
- **Transparent Failure**: Missing packages must raise standard import failures immediately at process startup so environment misconfigurations are surfaced and fixed, rather than silently degrading into fallback behavior.

---

## Rule 3: Never Use `TYPE_CHECKING` to Avoid Circular Dependencies
- **No `if TYPE_CHECKING:` Workarounds**: Do not use `typing.TYPE_CHECKING` or forward-reference string annotations (`'ClassName'`) as a band-aid to dodge circular import errors.
- **Design Clean Architecture**: Circular imports indicate an architectural flaw where modules are coupled inappropriately. When a circular dependency arises:
  1. Extract shared domain models, schemas, and enums into a dedicated, decoupled domain layer (e.g., `core/models.py`).
  2. Extract abstract interfaces or protocols into separate base contracts (e.g., `vector_store/base.py`).
  3. Ensure dependency flow is strictly unidirectional (e.g., `Service -> Domain Model`, never bidirectional cycles).

---

## Rule 4: Remove Older Features When a Replacement is Chosen
- **Zero Vestigial Code**: When a new replacement is chosen and can fully replace an older implementation, decisively delete the old feature, obsolete functions, and dead logic.
- **Do Not Maintain Dual Paths**: Do not leave old implementations alongside new ones as redundant fallbacks or commented-out blocks. Commit fully to the chosen modern implementation.

---

## Rule 5: Never Support Backward Compatibility; Write Clean Code Instead
- **No Legacy Compatibility Bloat**: Never compromise code quality or add complexity, deprecation wrappers, or compatibility shims to support legacy interfaces or obsolete behaviors.
- **Write Clean, Direct Code**: Always write clean, idiomatic, and modern code targeting the current architecture. Refactor callers and consumers directly rather than maintaining backward-compatible layers.
