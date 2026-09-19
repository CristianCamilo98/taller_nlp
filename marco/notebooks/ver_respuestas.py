"""Ver la respuesta completa del agente."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from responder import responder

r = responder("¿Cuál fue el revenue de NVIDIA en FY2024?")
print(json.dumps(r, indent=2, ensure_ascii=False, default=str))