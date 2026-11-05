"""Fast sanity tests (stdlib unittest; run: python -m unittest discover -s tests)."""
import unittest

import numpy as np

from src import config
from src.data_generation import generate_dataset
from src.features import HandcraftedFeatures, shannon_entropy
from src.models import MODEL_NAMES, build_model
from src.preprocessing import canonicalize
from src.windowed import WindowedDetector, split_windows


class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_dataset(n_benign=400, n_sqli=200, n_xss=200)

    def test_counts_and_labels(self):
        self.assertEqual(self.df.label.value_counts().to_dict(), {"benign": 400, "sqli": 200, "xss": 200})
        self.assertTrue(self.df.text.is_unique)

    def test_deterministic(self):
        again = generate_dataset(n_benign=400, n_sqli=200, n_xss=200)
        self.assertTrue((again.text.values == self.df.text.values).all())

    def test_obfuscations_known(self):
        known = set(config.SEEN_OBFUSCATIONS) | set(config.HELD_OUT_OBFUSCATIONS)
        self.assertTrue(set(self.df.obfuscation) <= known)

    def test_only_reserved_hosts(self):
        # payload hosts must use reserved example/test TLDs
        xss = self.df[self.df.label == "xss"].text
        self.assertFalse(xss.str.contains(r"src=//(?!evil\.example|attacker\.test|cdn\.example|x\.test)", regex=True).any())


class TestPreprocessing(unittest.TestCase):
    def test_double_url(self):
        self.assertEqual(canonicalize("%2527%2520OR%25201%253D1"), "' or 1=1")

    def test_entities_and_js_escapes(self):
        self.assertEqual(canonicalize("&#x3c;svg&#62;"), "<svg>")
        self.assertEqual(canonicalize("\\u003cimg\\u003e"), "<img>")

    def test_comments(self):
        self.assertEqual(canonicalize("1/**/UNION/**/SELECT"), "1 union select")
        self.assertEqual(canonicalize("1 /*!50000UNION*/ SELECT"), "1 union select")


class TestFeatures(unittest.TestCase):
    def test_shape_and_names(self):
        hf = HandcraftedFeatures()
        X = hf.transform(["hello", "' OR 1=1--", "<img src=x onerror=alert(1)>"])
        self.assertEqual(X.shape, (3, len(hf.feature_names_)))
        names = list(hf.feature_names_)
        self.assertEqual(X[1, names.index("has_tautology")], 1.0)
        self.assertEqual(X[2, names.index("has_event_handler")], 1.0)

    def test_entropy(self):
        self.assertAlmostEqual(shannon_entropy("aaaa"), 0.0)
        self.assertAlmostEqual(shannon_entropy("ab"), 1.0)


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate_dataset(n_benign=600, n_sqli=300, n_xss=300)

    def test_all_models_fit_predict(self):
        for key in MODEL_NAMES:
            m = build_model(key).fit(self.df.text, self.df.label)
            pred = m.predict(["' UNION SELECT username,password FROM users--",
                              "<script>alert(1)</script>", "cheap red shoes"])
            self.assertEqual(len(pred), 3, key)
            self.assertTrue(set(pred) <= set(config.LABELS), key)

    def test_windowed(self):
        m = build_model("tfidf_lr").fit(self.df.text, self.df.label)
        det = WindowedDetector(m, 48, 16)
        p = det.predict(["great shoes " * 20 + "<svg onload=alert(1)>" + " thanks" * 20])
        self.assertEqual(p[0], "xss")
        self.assertGreater(len(split_windows("x" * 200, 48, 16)), 1)
        self.assertTrue(np.isfinite(det.malicious_score(["hi"])).all())


if __name__ == "__main__":
    unittest.main()
