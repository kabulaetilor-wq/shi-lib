import unittest

from shi.ingestion.masking import mask, mask_report
from shi.ingestion.normalize import main_service, max_severity, parse_logs


class TestMasking(unittest.TestCase):
    def test_secrets_are_masked(self):
        text = (
            "api_key=sk_live_abcdef123456789 password=hunter2 Authorization: Bearer abcdefghijklmnop\n"
            "aws AKIAABCDEFGHIJKLMNOP gh ghp_abcdefghijklmnopqrstuvwxyz0123 url postgres://admin:s3cret@db/x"
        )
        out = mask(text)
        for leaked in ("sk_live_abcdef", "hunter2", "abcdefghijklmnop", "AKIAABCDEFGHIJKLMNOP", "ghp_abcdef", "s3cret"):
            self.assertNotIn(leaked, out)

    def test_private_key_block(self):
        out = mask("x\n-----BEGIN RSA PRIVATE KEY-----\nMIIabc\ndef\n-----END RSA PRIVATE KEY-----\ny")
        self.assertNotIn("MIIabc", out)
        self.assertIn("[CLE_PRIVEE]", out)

    def test_pii_email_and_card(self):
        out, counts = mask_report("contact a.b@example.com carte 4111 1111 1111 1111 ok")
        self.assertIn("[EMAIL]", out)
        self.assertIn("[CARTE]", out)
        self.assertEqual(counts["email"], 1)

    def test_non_card_number_untouched(self):
        self.assertEqual(mask("commande 1234567890123"), "commande 1234567890123")

    def test_normal_text_unchanged(self):
        s = "Le serveur ne répond plus depuis 10h"
        self.assertEqual(mask(s), s)


class TestNormalize(unittest.TestCase):
    LOG = (
        "2026-10-17 10:02:11 ERROR billing/prod: échec paiement\n"
        "Traceback (most recent call last):\n"
        '  File "app.py", line 3, in run\n'
        "2026-10-17T10:02:12Z [WARN] lent\n"
        "\n"
        "INFO tout va bien\n"
    )

    def test_parse(self):
        ev = parse_logs(self.LOG, incident_id="inc-1")
        self.assertEqual(len(ev), 3)
        self.assertEqual((ev[0].service, ev[0].environment, ev[0].severity), ("billing", "prod", "ERROR"))
        self.assertIn("Traceback", ev[0].message)  # rattaché à l'événement précédent
        self.assertEqual(ev[1].severity, "WARNING")
        self.assertEqual(ev[2].service, "inconnu")
        self.assertEqual(ev[0].incident_id, "inc-1")

    def test_helpers(self):
        ev = parse_logs(self.LOG)
        self.assertEqual(max_severity(ev), "ERROR")
        self.assertEqual(main_service(ev), "billing")
        self.assertEqual(max_severity([]), "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
