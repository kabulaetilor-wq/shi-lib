import unittest

try:
    from fastapi.testclient import TestClient
    from shi.api.main import create_app
    HAVE_API = True
except ImportError:  # FastAPI/httpx absents (ex. Pydroid 3) : tests ignorés
    HAVE_API = False

from tests.helpers import DESC, LOGS, RepoTestCase


@unittest.skipUnless(HAVE_API, "fastapi/httpx non installés")
class TestApi(RepoTestCase):
    KEY = "cle-de-test"

    def setUp(self):
        super().setUp()
        self.client = TestClient(create_app(self.KEY, [self.tmp]))
        self.h = {"X-API-Key": self.KEY}

    def test_auth_required(self):
        self.assertEqual(self.client.get("/api/audit/verify").status_code, 401)
        self.assertEqual(self.client.get("/api/audit/verify", headers={"X-API-Key": "mauvaise"}).status_code, 401)
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_repo_outside_allowed_roots_rejected(self):
        r = self.client.post("/api/projects", headers=self.h, json={"name": "x", "repo_path": "/etc"})
        self.assertEqual(r.status_code, 400)

    def test_destructive_auto_rejected(self):
        r = self.client.post("/api/projects", headers=self.h, json={
            "name": "x", "repo_path": str(self.repo), "mode": "auto", "auto_allowed": ["destructive"]})
        self.assertEqual(r.status_code, 422)

    def test_manual_flow(self):
        h = self.h
        pid = self.client.post("/api/projects", headers=h, json={
            "name": "demo", "repo_path": str(self.repo), "granted": ["apply_fix"]}).json()["id"]
        iid = self.client.post("/api/projects/%s/incidents" % pid, headers=h,
                               json={"description": DESC, "logs": LOGS}).json()["id"]
        self.assertEqual(self.client.post("/api/incidents/%s/analyze" % iid, headers=h).status_code, 202)
        inc = self.client.get("/api/incidents/%s" % iid, headers=h).json()
        self.assertEqual(inc["status"], "analyse")
        sols = self.client.post("/api/incidents/%s/solutions" % iid, headers=h).json()["solutions"]
        sol = next(s for s in sols if s["patch"])
        self.assertNotIn("new_source", sol["patch"])
        # approuver avant test : refusé
        self.assertEqual(self.client.post("/api/solutions/%s/approve" % sol["id"], headers=h, json={"approver": "alice"}).status_code, 409)
        self.client.post("/api/solutions/%s/test" % sol["id"], headers=h)
        ok = self.client.post("/api/solutions/%s/approve" % sol["id"], headers=h, json={"approver": "alice"})
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(ok.json()["status"], "resolu")
        report = self.client.get("/api/incidents/%s/report" % iid, headers=h).json()["markdown"]
        self.assertIn("RÉUSSITE", report)
        self.assertNotIn("hunter2", report)
        self.assertTrue(self.client.get("/api/audit/verify", headers=h).json()["valid"])

    def test_auto_endpoint_forbidden_in_manual_project(self):
        pid = self.client.post("/api/projects", headers=self.h, json={"name": "d", "repo_path": str(self.repo)}).json()["id"]
        iid = self.client.post("/api/projects/%s/incidents" % pid, headers=self.h, json={"description": DESC}).json()["id"]
        self.assertEqual(self.client.post("/api/incidents/%s/auto" % iid, headers=self.h).status_code, 403)


if __name__ == "__main__":
    unittest.main()
