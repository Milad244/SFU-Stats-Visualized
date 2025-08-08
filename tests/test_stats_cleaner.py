import unittest
from sfu_stats import get_all_cleaned_stats
from sfu_stats import StatCategory
import pandas as pd


class TestGetAllCleanedSFUStats(unittest.TestCase):
    """
    Tests for get_all_cleaned_stats
    """

    def test_stats_were_gotten(self):
        """
        Testing that get_all_cleaned_stats returns the correct data structure and non-empty values.
        :return:
        """
        result = get_all_cleaned_stats()
        self.assertIsInstance(result, dict)
        self.assertTrue(len(result) > 0)
        for _, stat_category in result.items():
            self.assertIsInstance(stat_category, StatCategory)
            for __, stat in stat_category.stats.items():
                self.assertIsInstance(stat.df, pd.DataFrame)
                self.assertFalse(stat.df.empty)
