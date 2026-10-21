import unittest
from examples.libero.selection import select_tasks, select_initial_states


class LiberoSelection(unittest.TestCase):
    def test_nonfirst_task_and_fixed_states(self):
        self.assertEqual(select_tasks(90, [7]), [7])
        self.assertEqual(select_initial_states(50, 3, [41, 2, 17]), [41, 2, 17])

    def test_default_protocol_is_preserved(self):
        self.assertEqual(select_tasks(10), list(range(10)))
        self.assertEqual(select_initial_states(50, 50), list(range(50)))

    def test_invalid_diagnostics_are_rejected(self):
        for ids in ([], [7, 7], [-1], [90]):
            with self.assertRaises(ValueError):
                select_tasks(90, ids)
        with self.assertRaises(ValueError):
            select_tasks(90, [7], 1)
        for trials, ids in ((2, [1]), (2, [1, 1]), (1, [50]), (51, None)):
            with self.assertRaises(ValueError):
                select_initial_states(50, trials, ids)


if __name__ == "__main__":
    unittest.main()
