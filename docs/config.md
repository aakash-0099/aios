# Configuration

AIOS loads its configuration from `aios/config/config.yaml` when
`get_settings()` is first called. Environment variables override YAML values.
The returned `AIOSSettings` object is frozen; its fields are accessed as
attributes, for example `get_settings().storage_root`.

Environment overrides use uppercase `AIOS_` names. Top-level fields use
`AIOS_<FIELD>`; section fields use `AIOS_<SECTION>_<KEY>`. For example,
`llm.provider` is overridden with `AIOS_LLM_PROVIDER`. The older short names
`AIOS_REQUEST_TIMEOUT` and `AIOS_CONNECT_TIMEOUT` are also accepted for the
timeout fields; the section-based names take precedence if both are set.

| YAML key | Attribute | Default | Environment variable |
| --- | --- | --- | --- |
| `environment` | `environment` | `development` | `AIOS_ENVIRONMENT` |
| `debug` | `debug` | `true` | `AIOS_DEBUG` |
| `log_level` | `log_level` | `INFO` | `AIOS_LOG_LEVEL` |
| `queue.backend` | `queue_backend` | `memory` | `AIOS_QUEUE_BACKEND` |
| `llm.provider` | `llm_provider` | `openai` | `AIOS_LLM_PROVIDER` |
| `storage.root` | `storage_root` | `./data/storage` | `AIOS_STORAGE_ROOT` |
| `timeouts.request` | `request_timeout` | `30` seconds | `AIOS_TIMEOUTS_REQUEST` (alias: `AIOS_REQUEST_TIMEOUT`) |
| `timeouts.connect` | `connect_timeout` | `10` seconds | `AIOS_TIMEOUTS_CONNECT` (alias: `AIOS_CONNECT_TIMEOUT`) |

`queue.backend` accepts `memory` or `redis`. `llm.provider` accepts `openai`,
`groq`, `ollama`, or `mock`. Timeouts must be positive integers, `debug` must
be a boolean, and `log_level` must be a standard Python logging level. Unknown
YAML keys and invalid values raise `ConfigurationError`.

```python
from aios.config import get_settings

settings = get_settings()
print(settings.storage_root)
```