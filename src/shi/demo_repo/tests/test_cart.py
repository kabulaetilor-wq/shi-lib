import unittest

from cart import add_item, total


class TestCart(unittest.TestCase):
    def test_add_item_single(self):
        self.assertEqual(add_item("pomme"), ["pomme"])

    def test_carts_are_independent(self):
        add_item("pomme")
        self.assertEqual(add_item("poire"), ["poire"])

    def test_total(self):
        self.assertEqual(total([1, 2, 3]), 6)


if __name__ == "__main__":
    unittest.main()
