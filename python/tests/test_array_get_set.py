import unittest
import mlx.core as mx
import mlx_tests


class TestArrayGetSet(unittest.TestCase):
    def test_get_scalar_and_slice(self):
        x = mx.array([[1, 2], [3, 4]])
        self.assertEqual(x.get((0, 1)).item(), 2)

        y = mx.arange(12).reshape(3, 4)
        sub = y.get((slice(None), slice(1, 3)))
        self.assertEqual(sub.tolist(), [[1, 2], [5, 6], [9, 10]])

    def test_set_scalar_and_slice(self):
        x = mx.array([[1, 2], [3, 4]])
        x.set((1, 0), 99)
        self.assertEqual(x.tolist(), [[1, 2], [99, 4]])

        y = mx.zeros((3, 4), dtype=mx.int32)
        y.set((slice(0, 2), slice(1, 3)), 7)
        self.assertEqual(
            y.tolist(),
            [
                [0, 7, 7, 0],
                [0, 7, 7, 0],
                [0, 0, 0, 0],
            ],
        )

    def test_fancy_indexing_and_mask(self):
        y = mx.arange(12).reshape(3, 4)
        rows = mx.array([0, 2])
        cols = mx.array([3, 1])
        got = y.get((rows, cols))
        self.assertEqual(got.tolist(), [3, 9])

        mask = (y % 2) == 0
        evens = y.get(mask)
        self.assertTrue(all(v % 2 == 0 for v in evens.tolist()))

    def test_set_with_array_value_shape_broadcast(self):
        y = mx.zeros((3, 4), dtype=mx.int32)
        y.set((1, slice(None)), mx.array([1, 2, 3, 4]))
        self.assertEqual(y.tolist()[1], [1, 2, 3, 4])

    def test_negative_indices_and_steps(self):
        y = mx.arange(10)
        self.assertEqual(y.get((-1,)).item(), 9)
        self.assertEqual(y.get((slice(None, None, -1),)).tolist(), list(range(9, -1, -1)))


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()