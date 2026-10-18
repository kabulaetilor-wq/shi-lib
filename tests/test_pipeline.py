import unittest

from shi.audit.log import AuditLog
from shi.execution.pipeline import Pipeline, PipelineError
from shi.reports.report import render_report
from shi.sandbox.runner import SandboxError, SandboxRunner
from shi.config import Limits
from shi.security.guard import DEFAULT_GRANTED, Mode, Policy
from tests.helpers import RepoTestCase, policy


class TestSandbox(RepoTestCase):
    def test_baseline_reproduces_bug_and_original_untouched(self):
        r = SandboxRunner()
        with r.workspace(self.repo) as ws:
            res = r.run_tests(ws)
            self.assertEqual((res.passed, res.failed), (2, 1))
            self.assertFalse(res.ok)
            self.assertIn("reseau", res.isolation)
        self.assertEqual((self.repo / "cart.py").read_text().count("None"), 0)

    def test_timeout_kills_process(self):
        (self.repo / "tests" / "test_slow.py").write_text(
            "import unittest, time\nclass T(unittest.TestCase):\n    def test_x(self):\n        time.sleep(30)\n")
        r = SandboxRunner(Limits(wall_seconds=2))
        with r.workspace(self.repo) as ws:
            res = r.run_tests(ws)
        self.assertTrue(res.timed_out)
        self.assertLess(res.duration, 10)

    def test_path_escape_rejected(self):
        r = SandboxRunner()
        with r.workspace(self.repo) as ws:
            with self.assertRaises(SandboxError):
                r.write_files(ws, {"../evil.py": "x"})

    def test_symlinks_not_followed(self):
        import os
        secret = self.tmp / "secret.txt"
        secret.write_text("TOP SECRET")
        os.symlink(secret, self.repo / "link.txt")
        r = SandboxRunner()
        with r.workspace(self.repo) as ws:
            self.assertFalse((ws / "link.txt").exists())


class TestManualFlow(RepoTestCase):
    def test_full_manual_cycle(self):
        pipe = self.pipeline()
        inc = self.incident(pipe)
        self.assertNotIn("hunter2", inc.logs)
        self.assertNotIn("sk_live_abcdef", inc.logs)
        pipe.analyze(inc)
        self.assertEqual(inc.diagnosis.hypotheses[0].id, "etat_partage")
        self.assertFalse(inc.diagnosis.uncertain)
        self.assertTrue(len(inc.solutions) >= 3)  # plusieurs options comparées
        sol = next(s for s in inc.solutions if s.patch)
        self.assertEqual(sol.patch.rule, "mutable_default")
        plan = inc.plans[sol.id]
        self.assertIn("approbation requise", [s.decision for s in plan])

        pipe.test_solution(inc, sol.id)
        self.assertEqual(sol.status, "teste_ok")
        self.assertEqual(inc.status, "en_attente_approbation")
        self.assertNotIn("is None", (self.repo / "cart.py").read_text())  # rien modifié sans accord

        pipe.approve(inc, sol.id, "alice")
        self.assertEqual(inc.status, "resolu")
        self.assertIn("is None", (self.repo / "cart.py").read_text())
        self.assertEqual(inc.verification["outcome"], "reussite")

        pipe.rollback(inc)
        self.assertNotIn("is None", (self.repo / "cart.py").read_text())
        self.assertEqual(inc.status, "retour_arriere")

        report = render_report(inc, pipe.audit.verify())
        self.assertIn("Contenu suspect ignoré", report)  # le README hostile est signalé, pas obéi
        self.assertNotIn("hunter2", report)
        self.assertTrue(pipe.audit.verify())

    def test_cannot_approve_untested(self):
        pipe = self.pipeline()
        inc = pipe.analyze(self.incident(pipe))
        sol = next(s for s in inc.solutions if s.patch)
        with self.assertRaises(PipelineError):
            pipe.approve(inc, sol.id, "alice")

    def test_read_only_policy_blocks_apply(self):
        pipe = self.pipeline(apply_fix=False)
        inc = pipe.analyze(self.incident(pipe))
        sol = next(s for s in inc.solutions if s.patch)
        pipe.test_solution(inc, sol.id)
        with self.assertRaises(PermissionError):
            pipe.approve(inc, sol.id, "alice")
        self.assertNotIn("is None", (self.repo / "cart.py").read_text())

    def test_stale_file_is_detected(self):
        pipe = self.pipeline()
        inc = pipe.analyze(self.incident(pipe))
        sol = next(s for s in inc.solutions if s.patch)
        (self.repo / "cart.py").write_text((self.repo / "cart.py").read_text() + "\n# modifié\n")
        pipe.test_solution(inc, sol.id)
        self.assertEqual(sol.status, "obsolete")

    def test_non_automatable_solution_is_not_applied(self):
        pipe = self.pipeline()
        inc = pipe.analyze(self.incident(pipe))
        manual = next(s for s in inc.solutions if not s.patch)
        with self.assertRaises(PipelineError):
            pipe.approve(inc, manual.id, "alice")


class TestFailureHonesty(RepoTestCase):
    def test_failing_patch_is_never_deployed(self):
        # Le test exige un comportement que le correctif ne fournit pas.
        (self.repo / "tests" / "test_more.py").write_text(
            "import unittest\nfrom cart import add_item\nclass T(unittest.TestCase):\n"
            "    def test_impossible(self):\n        self.assertEqual(add_item('a'), ['z'])\n")
        pipe = self.pipeline()
        inc = pipe.analyze(self.incident(pipe))
        sol = next(s for s in inc.solutions if s.patch)
        pipe.test_solution(inc, sol.id)
        self.assertEqual(sol.status, "teste_echec")
        self.assertEqual(inc.status, "echec_tests")
        with self.assertRaises(PipelineError):
            pipe.approve(inc, sol.id, "alice")
        self.assertIn("ÉCHEC", render_report(inc))

    def test_unreproduced_bug_is_uncertain_not_success(self):
        (self.repo / "cart.py").write_text('def add_item(item, items=None):\n    return (items or []) + [item]\n\ndef total(p):\n    return sum(p)\n')
        pipe = self.pipeline()
        inc = pipe.analyze(self.incident(pipe))
        self.assertFalse([s for s in inc.solutions if s.patch])  # rien à corriger automatiquement
        self.assertNotEqual(inc.status, "resolu")


class TestAutoMode(RepoTestCase):
    def test_auto_fixes_low_risk_and_verifies(self):
        pipe = self.pipeline(auto=True)
        inc = pipe.run_auto(self.incident(pipe))
        self.assertEqual(inc.status, "resolu")
        self.assertIn("is None", (self.repo / "cart.py").read_text())
        self.assertTrue(pipe.audit.verify())

    def test_auto_refused_in_manual_mode(self):
        pipe = self.pipeline(auto=False)
        with self.assertRaises(PermissionError):
            pipe.run_auto(self.incident(pipe))

    def test_auto_stops_when_diagnosis_uncertain(self):
        pipe = self.pipeline(auto=True)
        inc = pipe.new_incident("p1", "Quelque chose ne va pas", "INFO a/b: ras\n", str(self.repo))
        pipe.run_auto(inc)
        self.assertEqual(inc.status, "intervention_requise")
        self.assertNotIn("is None", (self.repo / "cart.py").read_text())

    def test_auto_without_apply_permission_waits_for_approval(self):
        pol = Policy(Mode.AUTO, DEFAULT_GRANTED | {__import__("shi.security.guard", fromlist=["Action"]).Action.APPLY_FIX})
        pipe = Pipeline(pol, AuditLog())
        inc = pipe.run_auto(self.incident(pipe))
        self.assertEqual(inc.status, "en_attente_approbation")
        self.assertNotIn("is None", (self.repo / "cart.py").read_text())


if __name__ == "__main__":
    unittest.main()
