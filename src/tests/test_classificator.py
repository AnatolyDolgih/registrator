import context
import registrator.classification as cls 
import unittest

class TestClassificator(unittest.TestCase):
    def setUp(self):
        self.classificator = cls.Classificator("../models/intent_catcher.pt")
        
    def test_cls(self):
        self.assertEqual(self.classificator.bin_classification("Привет"), False)

