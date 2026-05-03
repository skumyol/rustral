# Contributing

Keep public APIs explicit and boring.

- Do not introduce global mutable model state.
- Do not let object drop order control computation semantics.
- Prefer small traits with narrow contracts.
- Keep backend-specific code out of model modules.
- Add tests for shape errors and boundary behavior.
- Parallelism must be observable at the runtime boundary.
