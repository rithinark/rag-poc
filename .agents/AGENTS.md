# Engineering Guidelines & Constraints

Follow these strict development rules across the entire codebase:

1. **Never Patch Actual Code to Support Unit Test Cases**:
   - Production code must never contain test-only conditionals, hacky shims, or bypass logic.
   - Use clean dependency injection, polymorphic driver interfaces, and proper test fixtures to exercise production code paths.

2. **Never Add Try/Catch Blocks for Imports**:
   - Never suppress missing libraries with `try ... except ImportError` blocks.
   - If a library or package is needed, explicitly declare it in `pyproject.toml` dependencies, install it, and use clean, direct top-level imports.

3. **Never Use `TYPE_CHECKING` to Avoid Circular Dependency Errors**:
   - Do not use `from typing import TYPE_CHECKING` or `if TYPE_CHECKING:` to sweep circular imports under the rug.
   - Resolve circular dependencies architecturally: refactor shared entities, value objects, and protocols into independent leaf layers with strictly unidirectional dependency flow.

4. **Never Hesitate to Remove Older Features When a Replacement is Chosen**:
   - Once a new solution or library replaces an older feature, delete the legacy code completely.
   - Do not leave dead code, obsolete pathways, or redundant dual-mode branches in the repository.

5. **Never Develop Supporting Backward Compatibility; Write Clean Code Instead**:
   - Do not write shims, adapters, or compatibility wrappers to maintain legacy interfaces.
   - Refactor callers directly and write clean, unambiguous, and modern code targeting the current standard.
