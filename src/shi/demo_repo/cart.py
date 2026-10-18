"""Panier d'un SaaS de vente en ligne (dépôt de démonstration SHI)."""


def add_item(item, items=[]):
    """Ajoute un article au panier et renvoie le panier."""
    items.append(item)
    return items


def total(prices):
    """Somme des prix."""
    return sum(prices)
