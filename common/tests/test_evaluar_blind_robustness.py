"""evaluar() debe sobrevivir a un JSONL de preguntas ciegas arbitrario: IDs
desconocidos, N preguntas cualquiera, y un fallo puntual (de responder() o
de los propios evaluadores, p. ej. por un esquema de pregunta inesperado)
no debe tirar abajo el resto del batch. Los resultados también se guardan
incrementalmente en guardar_en, no solo al final.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import common.evaluar as evaluator


def _base_response(**overrides):
    base = {
        "respuesta": "respuesta", "cifra": None, "unidad": None,
        "ticker": None, "ejercicio": None, "fuente": "ninguna",
        "cita": None, "chunk_id": None,
        "tool_calls_agente": [], "tool_calls_detallado": [],
        "tool_calls_bloqueados": [], "guardrail_retry_count": 0,
        "_telemetria": {},
    }
    base.update(overrides)
    return base


def _retry_meta():
    return {"retry_count": 0, "rate_limited": False, "backoff_s": 0.0,
            "latencia_activa_s": 0.0}


class BlindQuestionsRobustnessTests(unittest.TestCase):
    def _patched(self, **kwargs):
        patches = [
            patch.object(evaluator, "_sha256", return_value="hash"),
            patch.object(evaluator, "_commit_sha", return_value="commit"),
            patch.object(evaluator, "_artifact_hashes",
                        return_value={"corpus.faiss": "faiss"}),
            patch.object(evaluator, "_retrieval_final_provenance",
                        return_value={"faiss_sha256": "canonical"}),
        ]
        for key, value in kwargs.items():
            patches.append(patch.object(evaluator, key, value))
        return patches

    def _apply(self, patches):
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_unknown_question_ids_are_processed_generically(self):
        blind_questions = [
            {"id": "blind-alpha", "familia": "numerica", "pregunta": "p1",
             "ticker": "ZZZZ", "fiscal_year": 2099, "concept_xbrl": "Foo",
             "cifra_esperada": 1.0, "unidad": "USD"},
            {"id": "blind-beta", "familia": "extractiva", "pregunta": "p2",
             "ticker": "ZZZZ", "fiscal_year": 2099, "item_esperado": "1A"},
        ]
        self._apply(self._patched(
            cargar_golden=lambda _ruta: blind_questions,
            _invocar_con_reintentos=lambda pregunta, **kw: (
                _base_response(respuesta=f"respuesta a {pregunta}"),
                _retry_meta(),
            ),
        ))
        rows = evaluator.evaluar("blind10.jsonl", pausa_entre_preguntas=0)

        self.assertEqual(len(rows), 2)
        self.assertEqual([row["id"] for row in rows],
                         ["blind-alpha", "blind-beta"])
        self.assertIsNone(rows[0]["error"])

    def test_evaluator_exception_on_one_question_does_not_abort_the_batch(self):
        questions = [
            {"id": "q1", "familia": "numerica", "pregunta": "p1"},
            {"id": "q2", "familia": "numerica", "pregunta": "p2"},
            {"id": "q3", "familia": "numerica", "pregunta": "p3"},
        ]
        self._apply(self._patched(
            cargar_golden=lambda _ruta: questions,
            _invocar_con_reintentos=lambda pregunta, **kw: (
                _base_response(respuesta="ok", fuente="ninguna"),
                _retry_meta(),
            ),
        ))
        # evaluar_cifra_detallada revienta solo para la segunda pregunta,
        # simulando un esquema de blind question inesperado.
        def flaky_cifra(row, question):
            if question["id"] == "q2":
                raise KeyError("fiscal_year")
            return {"aplica": False, "acierto_cifra": None, "errores": []}

        with patch("common.eval.evaluador_cifra.evaluar_cifra_detallada",
                   side_effect=flaky_cifra):
            rows = evaluator.evaluar("blind10.jsonl", pausa_entre_preguntas=0)

        self.assertEqual(len(rows), 3)
        self.assertEqual([row["id"] for row in rows], ["q1", "q2", "q3"])
        self.assertIsNone(rows[0]["error"])
        self.assertIsNotNone(rows[1]["error"])
        self.assertIn("evaluator_error", rows[1]["error"])
        self.assertIsNone(rows[2]["error"])

    def test_results_are_saved_incrementally_not_only_at_the_end(self):
        questions = [
            {"id": f"q{i}", "familia": "numerica", "pregunta": f"p{i}"}
            for i in range(4)
        ]
        self._apply(self._patched(
            cargar_golden=lambda _ruta: questions,
            _invocar_con_reintentos=lambda pregunta, **kw: (
                _base_response(respuesta="ok"), _retry_meta(),
            ),
        ))
        with TemporaryDirectory() as tmp:
            destino = Path(tmp) / "sub" / "salida.jsonl"
            rows = evaluator.evaluar(
                "blind10.jsonl", guardar_en=str(destino),
                pausa_entre_preguntas=0,
            )
            lines = destino.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(len(lines), len(rows))
        saved_ids = [json.loads(line)["id"] for line in lines]
        self.assertEqual(saved_ids, [row["id"] for row in rows])

    def test_rerun_truncates_previous_output_instead_of_appending(self):
        questions = [{"id": "q1", "familia": "numerica", "pregunta": "p1"}]
        self._apply(self._patched(
            cargar_golden=lambda _ruta: questions,
            _invocar_con_reintentos=lambda pregunta, **kw: (
                _base_response(respuesta="ok"), _retry_meta(),
            ),
        ))
        with TemporaryDirectory() as tmp:
            destino = Path(tmp) / "salida.jsonl"
            destino.write_text("linea vieja de una corrida anterior\n",
                              encoding="utf-8")
            evaluator.evaluar("blind10.jsonl", guardar_en=str(destino),
                              pausa_entre_preguntas=0)
            lines = destino.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(len(lines), 1)
        self.assertNotIn("linea vieja", lines[0])

    def test_responder_failure_and_evaluator_failure_can_coexist(self):
        questions = [
            {"id": "q1", "familia": "numerica", "pregunta": "p1"},
            {"id": "q2", "familia": "extractiva", "pregunta": "p2"},
        ]

        def flaky_responder(pregunta, **kw):
            if pregunta == "p1":
                raise evaluator.InvocationFailed(
                    RuntimeError("fallo simulado del agente"), _retry_meta())
            return _base_response(respuesta="ok"), _retry_meta()

        self._apply(self._patched(
            cargar_golden=lambda _ruta: questions,
            _invocar_con_reintentos=flaky_responder,
        ))
        rows = evaluator.evaluar("blind10.jsonl", pausa_entre_preguntas=0)

        self.assertEqual(len(rows), 2)
        self.assertIsNotNone(rows[0]["error"])
        self.assertIsNone(rows[1]["error"])


if __name__ == "__main__":
    unittest.main()
