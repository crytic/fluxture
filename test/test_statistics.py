from unittest import TestCase

from fluxture.statistics import Statistics


class StatisticsTest(TestCase):
    def test_memoization(self):
        stats = Statistics((1, 2, 3, 4, 5))
        self.assertEqual(stats.average, sum((1, 2, 3, 4, 5)) / 5.0)
        self.assertEqual(stats.std_dev, stats.std_dev)

    def test_median(self):
        stats = Statistics((1, 2, 3, 4, 5))
        self.assertEqual(stats.median, 3)
        stats = Statistics((1, 2, 3, 4, 5, 6))
        self.assertEqual(stats.median, (3 + 4) / 2.0)
