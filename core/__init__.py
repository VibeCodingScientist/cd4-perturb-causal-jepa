"""core — frozen shared foundation for the CD4+ perturbation pipeline.

Import `core.contract` for the shared interface (paths, schemas, constants). Heavy
submodules (data, features, models) are imported explicitly by callers, never here,
so that `import core.contract` stays stdlib-light.
"""

__all__ = ["contract"]
