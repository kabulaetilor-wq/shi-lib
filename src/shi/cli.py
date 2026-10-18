"""Ligne de commande SHI.

    python -m shi.cli build-dataset
    python -m shi.cli train-classifier
    python -m shi.cli classify "Les pages chargent très lentement"
    python -m shi.cli train-transformer --steps 600
    python -m shi.cli generate "Le serveur"
    python -m shi.cli demo [--auto] [--approve]
    python -m shi.cli serve
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from shi.config import ARTIFACTS_DIR, DATASETS_DIR, DEMO_REPO

DEMO_DESCRIPTION = "Les paniers de deux clients sont mélangés sur le service panier"
DEMO_LOGS = """\
2026-10-17 10:02:11 ERROR panier/prod: Les articles d'un client apparaissent dans le panier d'un autre client
2026-10-17 10:02:12 WARN panier/prod: Le panier du client 4417 contient 12 articles inattendus
2026-10-17 10:03:40 ERROR panier/prod: AssertionError: expected ['poire'] got ['pomme', 'poire']
2026-10-17 10:03:41 INFO panier/prod: appel API avec api_key=sk_live_abcdef123456789 password=hunter2 contact=client@example.com
"""


def cmd_build_dataset(args):
    from shi.models.classifier import dataset

    data = dataset.build(args.seed)
    assert dataset.check_no_overlap(*[[r["text"] for r in rows] for rows in data.values()])
    DATASETS_DIR.mkdir(exist_ok=True)
    for split, rows in data.items():
        dataset.write_jsonl(DATASETS_DIR / ("%s.jsonl" % split), rows)
        print("%-5s %5d exemples" % (split, len(rows)))


def _ensure_dataset():
    if not (DATASETS_DIR / "train.jsonl").exists():
        cmd_build_dataset(argparse.Namespace(seed=0))


def cmd_train_classifier(args):
    """Entraîne SHI-0.2 sur le corpus rédigé à la main (12 catégories) et l'évalue sur un test jamais vu."""
    from shi.models.classifier import SHIClassifier, classification_report, format_report
    from shi.models.classifier.corpus_libre import CORPUS, N_TRAIN
    from shi.models.classifier.corpus_extra import EXTRA
    from shi.models.classifier.dataset import load_jsonl

    texts, labels = [], []
    for lab, rows in CORPUS.items():           # 22 premières phrases de chaque catégorie (les 8 autres = test de développement)
        texts += rows[:N_TRAIN]; labels += [lab] * N_TRAIN
    for lab, rows in EXTRA.items():
        texts += rows; labels += [lab] * len(rows)
    for fx in ("test_aveugle.jsonl",):           # ancien test, devenu données de développement
        x, y = load_jsonl(DATASETS_DIR / fx); texts += x; labels += y
    vx, vy = texts[::6], labels[::6]
    tx = [t for i, t in enumerate(texts) if i % 6]; ty = [l for i, l in enumerate(labels) if i % 6]
    clf = SHIClassifier(epochs=args.epochs)
    hist = clf.fit(tx, ty, vx, vy)
    print("apprentissage : %d phrases, validation : %d, époques : %d" % (len(tx), len(vx), len(hist["train_loss"])))
    xs, ys = load_jsonl(DATASETS_DIR / "test_final.jsonl")
    ps = clf.predict_labels(xs)
    print("\n== TEST FINAL (%d phrases écrites à part, jamais utilisées pour entraîner ni régler) ==" % len(xs))
    print(format_report(classification_report(ys, ps, clf.labels)))
    print("exactitude : %.3f" % (sum(a == b for a, b in zip(ps, ys)) / len(ys)))
    for t, y, p in zip(xs, ys, ps):
        if y != p:
            print("  erreur : %s -> %s (attendu %s)" % (t, p, y))
    clf.save(ARTIFACTS_DIR / "classifier")
    print("\nmodèle enregistré dans", ARTIFACTS_DIR / "classifier")
    print("Rappel : phrases rédigées par une seule personne, en un seul style. Ce score ne garantit rien sur de vrais incidents.")


def cmd_classify(args):
    from shi.models.classifier import load_default_classifier

    clf = load_default_classifier()
    if clf is None:
        sys.exit("Aucun modèle : lancer d'abord « train-classifier ».")
    p = clf.predict(" ".join(args.text))
    top = sorted(p.probs.items(), key=lambda kv: -kv[1])[:3]
    print("catégorie probable : %s (confiance %.2f)%s" % (p.label, p.confidence, "  [incertain]" if p.uncertain else ""))
    print("  " + ", ".join("%s %.2f" % kv for kv in top))
    print("Ne trouve pas seul la cause exacte : à confirmer par les preuves.")


def cmd_train_transformer(args):
    from shi.models.classifier.dataset import load_jsonl
    from shi.models.transformer import CharTokenizer, MiniTransformer

    _ensure_dataset()
    texts, _ = load_jsonl(DATASETS_DIR / "train.jsonl")
    corpus = "\n".join(texts)
    tok = CharTokenizer.from_text(corpus)
    data = tok.encode(corpus)
    cut = int(len(data) * 0.9)
    model = MiniTransformer(tok.size, context=args.context, dim=args.dim)
    print("corpus : %d caractères, vocabulaire : %d, perte de départ ≈ %.2f" % (len(data), tok.size, model.evaluate(data[cut:])))
    model.train(data[:cut], steps=args.steps, batch=16, log_every=max(args.steps // 10, 1))
    print("perte sur données non vues : %.3f" % model.evaluate(data[cut:]))
    model.save(ARTIFACTS_DIR / "transformer", tok)
    print("modèle enregistré dans", ARTIFACTS_DIR / "transformer")


def cmd_generate(args):
    from shi.models.transformer import MiniTransformer

    d = ARTIFACTS_DIR / "transformer"
    if not (d / "meta.json").exists():
        sys.exit("Aucun modèle : lancer d'abord « train-transformer ».")
    model, tok = MiniTransformer.load(d)
    ids = model.generate(tok.encode(args.prompt), args.length, args.temperature, args.seed)
    print(tok.decode(ids))


def cmd_demo(args):
    from shi.audit.log import AuditLog
    from shi.execution.pipeline import Pipeline
    from shi.models.classifier import load_default_classifier
    from shi.reports.report import render_report
    from shi.security.guard import Action, Mode, Policy, DEFAULT_GRANTED

    tmp = Path(tempfile.mkdtemp(prefix="shi-demo-"))
    repo = tmp / "repo"
    shutil.copytree(DEMO_REPO, repo, ignore=shutil.ignore_patterns("__pycache__"))
    auto = args.auto
    policy = Policy(
        mode=Mode.AUTO if auto else Mode.MANUAL,
        granted=DEFAULT_GRANTED | {Action.APPLY_FIX},
        auto_allowed=frozenset({Action.APPLY_FIX}) if auto else frozenset(),
    )
    pipe = Pipeline(policy, AuditLog(tmp / "audit.jsonl"), classifier=load_default_classifier())
    inc = pipe.new_incident("demo", DEMO_DESCRIPTION, DEMO_LOGS, str(repo))
    if auto:
        pipe.run_auto(inc)
    else:
        pipe.analyze(inc)
        best = next((s for s in inc.solutions if s.patch), None)
        if best:
            pipe.test_solution(inc, best.id)
            if args.approve and best.status == "teste_ok":
                pipe.approve(inc, best.id, "demo")
            elif best.status == "teste_ok":
                inc.decision = "Correctif validé en test. En attente de votre approbation (relancer avec --approve)."
    print(render_report(inc, pipe.audit.verify()))
    print("Dépôt de démonstration (copie jetable) :", repo)


def cmd_serve(args):
    import uvicorn

    from shi.api.main import create_app

    key = os.environ.get("SHI_API_KEY")
    if not key:
        sys.exit("Définir SHI_API_KEY (clé d'API) avant de démarrer le serveur.")
    roots = [Path(p).resolve() for p in os.environ.get("SHI_ALLOWED_ROOTS", "").split(os.pathsep) if p]
    if not roots:
        sys.exit("Définir SHI_ALLOWED_ROOTS (dossiers autorisés pour les dépôts, séparés par '%s')." % os.pathsep)
    uvicorn.run(create_app(key, roots), host=args.host, port=args.port)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="shi", description="SHI — SaaS H Intelligence")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build-dataset"); p.add_argument("--seed", type=int, default=0); p.set_defaults(fn=cmd_build_dataset)
    p = sub.add_parser("train-classifier"); p.add_argument("--epochs", type=int, default=400); p.set_defaults(fn=cmd_train_classifier)
    p = sub.add_parser("classify"); p.add_argument("text", nargs="+"); p.set_defaults(fn=cmd_classify)
    p = sub.add_parser("train-transformer")
    p.add_argument("--steps", type=int, default=600); p.add_argument("--context", type=int, default=48)
    p.add_argument("--dim", type=int, default=32); p.set_defaults(fn=cmd_train_transformer)
    p = sub.add_parser("generate"); p.add_argument("prompt"); p.add_argument("--length", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8); p.add_argument("--seed", type=int, default=0); p.set_defaults(fn=cmd_generate)
    p = sub.add_parser("demo"); p.add_argument("--auto", action="store_true"); p.add_argument("--approve", action="store_true"); p.set_defaults(fn=cmd_demo)
    p = sub.add_parser("serve"); p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8000); p.set_defaults(fn=cmd_serve)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
