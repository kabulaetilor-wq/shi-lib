"""Connaissance du domaine écrite à la main : indices de vocabulaire par catégorie.

Ce lexique est une règle explicite (pas un apprentissage). Il ajoute au classificateur des
indicateurs « ce texte contient du vocabulaire de la catégorie X ». Les motifs sont cherchés dans
le texte normalisé (minuscules, sans accents). Il est volontairement général : il ne doit pas être
ajusté phrase par phrase pour gagner des points de test."""
import re
from typing import Dict, List, Tuple

LEXICON: Dict[str, List[str]] = {
    "serveur": ["down", "hors ligne", "crash", "plant", "redemarr", "arret", "tombe", "injoignable", "indisponible",
                "oom", "segfault", "healthcheck", "pod ", "worker", "502", "503", "500", "5xx", "bad gateway",
                "ne repond plus", "ne demarre", "service unavailable", "mort", "panne"],
    "performance": ["lent", "lenteur", "latence", "cpu", "processeur", "memoire", "rame", "ramer", "lag", "swap",
                    "fuite", "garbage", "p95", "surcharg", "saturation", "pic de", "interminable", "eternite",
                    "temps de reponse", "response time", "load average", "poussif", "ralenti", "secondes"],
    "base_de_donnees": ["base de donnees", "sql", "postgres", "mysql", "mongo", "table", "deadlock", "verrou", "lock",
                        "index", "replica", "jdbc", "pool", "requete", "schema", "rollback automatique", "transaction",
                        "contrainte", "relation ", "enregistrement", "select"],
    "authentification": ["se connecter", "connexion impossible", "login", "logg", "mot de passe", "password", "token",
                         "jwt", "401", "403", "session", "oauth", "sso", "identif", "authentif", "credential",
                         "2fa", "double authentification", "compte", "droits d'acces", "droits d acces", "acces refuse",
                         "cle d'api", "cle d api", "cookie", "forbidden", "unauthorized"],
    "reseau": ["dns", "certificat", "ssl", "tls", "https", "domaine", "pare-feu", "pare feu", "firewall", "vpn",
               "proxy", "cdn", "ping", "paquet", "routage", "reseau", "smtp", "websocket", "load balancer",
               "504", "timeout reseau", "connection timed out", "unreachable", "registrar", "wi-fi", "4g"],
    "paiement": ["paiement", "payer", "paye", "carte", "stripe", "paypal", "facture", "factur", "rembours", "abonnement",
                 "checkout", "prelevement", "mobile money", "tva", "devise", "encaiss", "debite", "passerelle de paiement",
                 "payment", "virement", "reglement", "coupon", "panier"],
    "deploiement": ["deploi", "build", "pipeline", "release", "rollback", "docker", "gradle", "npm", "mise a jour",
                    "mise en production", "compilation", "livraison", "registry", "branche", "artefact", "ci ", " ci",
                    "github actions", "bleu-vert", "version", "migration de la release", "fusion"],
    "stockage": ["disque", "espace", "stockage", "fichier", "upload", "televers", "volume", "bucket", "sauvegarde",
                 "quota", "inode", "partition", "nas", "no space", "read-only", "plein", "capacite", "montage",
                 "piece jointe", "pieces jointes", "logs ont rempli", "rotation des logs"],
    "interface": ["bouton", "ecran", "affich", "formulaire", "mobile", "telephone", "css", "javascript", "menu",
                  "clic", "icone", "couleur", "responsive", "layout", "popup", "texte deborde", "image", "navigation",
                  "onglet", "clavier", "design", "page blanche", "mode sombre", "liste deroulante", "animation",
                  "alignement", "aligne", "safari", "tablette", "defiler"],
    "bug_applicatif": ["exception", "nullpointer", "typeerror", "indexerror", "keyerror", "calcul", "total", "montant",
                       "arrondi", "doublon", "bug", "division par zero", "none", "tri ", "filtre", "decale", "compteur",
                       "stock", "cas limite", "mauvais", "incorrect", "incoherent", "faux", "optionnel", "ecrase"],
    "securite": ["attaque", "faille", "vulnerab", "cve", "xss", "injection", "intrusion", "phishing", "malware",
                 "ransomware", "force brute", "ddos", "fuite de donnees", "secret", "suspect", "scann", "exploit",
                 "pirat", "malveill", "frauduleu", "antivirus", "divulgu", "expose", "publiquement", "en clair",
                 "ouvert a tous", "tentatives de connexion massives", "escalade de privileges", "/etc/passwd",
                 "robots", "bots"],
    "inconnu": ["bonjour", "bonsoir", "salut", "merci", "recette", "meteo", "film", "vacances", "cafe", "blague",
                "anniversaire", "match", "fete", "reunion", "conges", "capitale", "bonne annee", "heure ?"],
}

_COMPILED: List[Tuple[str, List[str]]] = [(c, [p.lower() for p in ps]) for c, ps in LEXICON.items()]


def lexicon_hits(normalized_text: str) -> List[str]:
    """Catégories dont au moins un motif apparaît dans le texte (déjà normalisé)."""
    t = " " + normalized_text + " "
    return [c for c, pats in _COMPILED if any(p in t for p in pats)]
