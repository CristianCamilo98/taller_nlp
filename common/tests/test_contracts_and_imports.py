from __future__ import annotations

import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class ImportTests(unittest.TestCase):
    def test_public_modules_import_offline(self):
        modules = [
            "common.responder", "common.evaluar", "common.agent",
            "common.tools", "common.eval.evaluador_cifra",
            "common.eval.evaluador_cita", "common.eval.evaluador_trayectoria",
        ]
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        for module in modules:
            completed = subprocess.run(
                [sys.executable, "-B", "-c", f"import {module}"],
                cwd=ROOT, env=environment, capture_output=True, text=True,
            )
            self.assertEqual(completed.returncode, 0,
                             f"{module}: {completed.stderr}")


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (ROOT / "common" / "tools" / "tools.py").read_text(
            encoding="utf-8")
        cls.tree = ast.parse(source)

    def _signature(self, name):
        node = next(n for n in self.tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        names = [arg.arg for arg in node.args.args]
        defaults = [ast.unparse(default) for default in node.args.defaults]
        return names, defaults

    def test_four_public_signatures(self):
        self.assertEqual(self._signature("list_available"), ([], []))
        self.assertEqual(self._signature("get_xbrl_fact")[0],
                         ["ticker", "fiscal_year", "concept"])
        self.assertEqual(self._signature("search_filings"),
                         (["query", "ticker", "fiscal_year", "item", "k"],
                          ["None", "None", "None", "5"]))
        self.assertEqual(self._signature("read_section")[0],
                         ["ticker", "fiscal_year", "item"])

    def test_schema_keeps_old_fields_and_adds_comparative_fields(self):
        tree = ast.parse((ROOT / "common" / "agent" / "schema.py").read_text(
            encoding="utf-8"))
        model = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        fields = {node.target.id for node in model.body
                  if isinstance(node, ast.AnnAssign)
                  and isinstance(node.target, ast.Name)}
        self.assertTrue({"respuesta", "cifra", "unidad", "ticker", "ejercicio",
                         "fuente", "cita", "chunk_id"}.issubset(fields))
        self.assertTrue({"concepto_xbrl", "ejercicio_inicial",
                         "ejercicio_final", "valor_inicial", "valor_final",
                         "delta", "porcentaje"}.issubset(fields))
