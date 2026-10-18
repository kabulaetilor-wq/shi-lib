import shutil
import tempfile
import unittest
from pathlib import Path

from shi.config import DEMO_REPO
from shi.execution.pipeline import Pipeline
from shi.security.guard import DEFAULT_GRANTED, Action, Mode, Policy

DESC = "Les paniers de deux clients sont mélangés sur le service panier"
LOGS = (
    "2026-10-17 10:02:11 ERROR panier/prod: Les articles d'un client apparaissent chez un autre client\n"
    "2026-10-17 10:02:12 WARN panier/prod: Le panier du client 4417 contient 12 articles inattendus\n"
    "2026-10-17 10:03:40 ERROR panier/prod: AssertionError: expected ['poire'] got ['pomme', 'poire']\n"
    "2026-10-17 10:03:41 INFO panier/prod: api_key=sk_live_abcdef123456789 password=hunter2\n"
)


def policy(auto=False, apply_fix=True):
    granted = DEFAULT_GRANTED | ({Action.APPLY_FIX} if apply_fix else set())
    return Policy(Mode.AUTO if auto else Mode.MANUAL, frozenset(granted),
                  frozenset({Action.APPLY_FIX}) if auto else frozenset())


class RepoTestCase(unittest.TestCase):
    """Copie jetable du dépôt de démonstration."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="shi-test-"))
        self.repo = self.tmp / "repo"
        shutil.copytree(DEMO_REPO, self.repo, ignore=shutil.ignore_patterns("__pycache__"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def pipeline(self, **kw):
        return Pipeline(policy(**kw))

    def incident(self, pipe):
        return pipe.new_incident("p1", DESC, LOGS, str(self.repo))
