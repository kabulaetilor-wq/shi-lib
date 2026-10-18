import ast
import unittest

from shi.analysis.code_analyzer import analyze_source, match_frames, parse_tracebacks
from shi.diagnosis.engine import diagnose
from shi.ingestion.normalize import parse_logs
from shi.solutions.patches import (fix_bare_except, fix_eq_none, fix_mutable_defaults,
                                   fix_requests_timeout, propose_patches)

MUTABLE = '''def add(item, items=[], *, opts={}):
    """doc"""
    items.append(item)
    return items


def other(x=1):
    return x
'''


class TestPatches(unittest.TestCase):
    def test_mutable_default(self):
        new = fix_mutable_defaults(MUTABLE)
        ast.parse(new)
        ns = {}
        exec(new, ns)  # code de test contrôlé
        self.assertEqual(ns["add"]("a"), ["a"])
        self.assertEqual(ns["add"]("b"), ["b"])
        self.assertIn('"""doc"""\n    if items is None:', new)
        self.assertIsNone(fix_mutable_defaults("def f(x=1):\n    return x\n"))

    def test_bare_except(self):
        new = fix_bare_except("try:\n    1/0\nexcept:\n    pass\n")
        self.assertIn("except Exception:", new)
        self.assertIsNone(fix_bare_except("try:\n    pass\nexcept ValueError:\n    pass\n"))

    def test_eq_none(self):
        new = fix_eq_none("if a == None or b != None:\n    pass\ns = 'x == None'\n")
        self.assertIn("if a is None or b is not None:", new)
        self.assertIn("'x == None'", new)  # les chaînes ne sont pas modifiées

    def test_requests_timeout(self):
        new = fix_requests_timeout("import requests\nr = requests.get(url)\nq = requests.post(u, data=d)\nz = requests.get(u, timeout=3)\n")
        self.assertIn("requests.get(url, timeout=10)", new)
        self.assertIn("requests.post(u, data=d, timeout=10)", new)
        self.assertEqual(new.count("timeout=10"), 2)
        self.assertIn("timeout=3", new)

    def test_propose_patches_have_diff_and_hash(self):
        ps = propose_patches("cart.py", MUTABLE)
        self.assertEqual([p.rule for p in ps], ["mutable_default"])
        self.assertIn("--- a/cart.py", ps[0].diff)
        self.assertEqual(len(ps[0].base_sha256), 64)


class TestAnalysis(unittest.TestCase):
    def test_findings(self):
        _, f = analyze_source(MUTABLE + "\ntry:\n    pass\nexcept:\n    pass\n")
        self.assertEqual({x.rule for x in f}, {"mutable_default", "bare_except"})

    def test_syntax_error_is_reported(self):
        self.assertEqual(analyze_source("def (:")[1][0].rule, "syntax_error")

    def test_traceback_matching(self):
        frames = parse_tracebacks('Traceback\n  File "/srv/app/cart.py", line 4, in add_item\n  File "/usr/lib/x.py", line 1, in y\n')
        self.assertEqual(match_frames(frames, ["cart.py", "other.py"]), [("cart.py", 4, "add_item")])


class TestDiagnosis(unittest.TestCase):
    def test_evidence_based_hypothesis(self):
        ev = parse_logs("ERROR svc/prod: connection refused on port 5432\nERROR svc/prod: connection refused on port 5432\n")
        d = diagnose("Le service ne répond plus", ev)
        self.assertEqual(d.hypotheses[0].id, "conn_refused")
        self.assertTrue(d.hypotheses[0].evidence)
        self.assertFalse(d.uncertain)

    def test_no_evidence_means_uncertain(self):
        d = diagnose("Quelque chose ne va pas", parse_logs("INFO a/b: ras\n"))
        self.assertTrue(d.uncertain)
        self.assertEqual(d.hypotheses, [])


if __name__ == "__main__":
    unittest.main()
