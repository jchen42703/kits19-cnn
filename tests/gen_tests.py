import unittest
import numpy as np

class SliceGeneratorTests(unittest.TestCase):
    """
    Simple test that unit-tests individual methods. Some are rewritten to be independent
    functions rather than methods because of how the generators require file paths. Really
    not trying to make random temporary files in this TestCase, but that might be an
    idea for the future.
    """
    def setUp(self):
        self.input_shape = (5, 1, 60, 224, 224)
        self.y = np.random.randint(0, 2, self.input_shape).astype(np.int32)
        self.y_blank = np.zeros(self.input_shape)
        self.both = np.vstack([self.y, self.y_blank]) # (5, 1, 120, 224, 224)

    def test_get_all_per_label_pos_slice_idx(self):
        """
        Test that slice indices all correspond to slices that have at least
        one non-background pixel.
        """
        def get_all_per_label_pos_slice_idx(label):
            """
            Gets a random positive slice index. Assumes the background class is 0.
            Args:
                label: numpy array with the dims (batch, n_channels, x,y,z)
            Returns:
                a list of all non-background class integer slice indices
            """
            pos_idx = np.nonzero(label)[2]
            return pos_idx.squeeze().tolist()
        # tests that it can handle blank labels
        all_blank = get_all_per_label_pos_slice_idx(self.y_blank)
        self.assertEqual(all_blank, [])
        # tests that it works properly for legit-er labels
        all_pos = list(set(get_all_per_label_pos_slice_idx(self.both)))
        for idx in all_pos:
            unique = np.unique(self.both[:, :, idx])
            assert unique.size > 1, "Outputted slices must have at least one \
                                     non-background label."
        print("Done!")
        self.assertTrue(True)

    def test_get_rand_pos_slice_idx(self):
        """
        Test that the outputted slice indices corresponds to a slices that has at least
        one non-background pixel.
        """
        def get_rand_pos_slice_idx(label):
            """
            Gets a random positive slice index. Assumes the background class is 0.
            Args:
                label: numpy array with the dims (batch, n_channels, x,y,z)
            Returns:
                an integer representing a random non-background class slice index
            """
            # "n_dims" numpy arrays of all possible positive pixel indices for the label
            slice_indices = np.nonzero(label)[2]
            # finding random positive class index
            random_pos_coord = np.random.choice(slice_indices)
            return random_pos_coord
        pos_idx = get_rand_pos_slice_idx(self.both)
        unique = np.unique(self.both[:, :, pos_idx])
        self.assertTrue(unique.size > 1)# "Outputted slices must have at least one \
                                        # non-background label."
if __name__ == "__main__":
    unittest.main()
