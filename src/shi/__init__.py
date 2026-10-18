"""SHI — SaaS H Intelligence : diagnostic d'incidents logiciels, en conseil seulement.

Utilisation rapide ::

    import shi
    r = shi.diagnose("Les clients ne peuvent plus payer", logs="ERROR stripe card_declined")
    print(r["category"], r["uncertain"])
    for h in r["hypotheses"]: print(h["label"], h["confidence"], h["evidence"])
    print(r["advice"])

La bibliothèque n'exécute aucune commande et n'écrit rien sur votre système par défaut.
Les secrets présents dans le texte sont masqués avant l'analyse.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

__version__ = "0.3.0"
__all__ = ["diagnose", "classify", "load_classifier", "SHIClassifier", "__version__"]

from shi.models.classifier import SHIClassifier, load_default_classifier  # noqa: E402

_cache: Dict[str, Optional[SHIClassifier]] = {}


def load_classifier() -> Optional[SHIClassifier]:
    """Charge (une seule fois) le classificateur livré avec le paquet, ou celui de SHI_HOME."""
    if "clf" not in _cache:
        _cache["clf"] = load_default_classifier()
    return _cache["clf"]


def classify(text: str):
    """Catégorie probable d'une description : objet avec label, confidence, probs, uncertain."""
    clf = load_classifier()
    if clf is None:
        raise RuntimeError("Aucun modèle disponible : lancez « python -m shi.cli train-classifier ».")
    return clf.predict(text)


def diagnose(description: str, logs: str = "") -> Dict[str, Any]:
    """Diagnostic en conseil seulement : catégorie, hypothèses avec preuves, vérifications sûres."""
    from shi.diagnosis.service import diagnose_text

    return diagnose_text(description, logs, load_classifier())
