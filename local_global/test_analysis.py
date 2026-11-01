"""Scientific sanity checks independent of the benchmark's observed rankings."""
import unittest

import numpy as np

from analyze_local_global import make_masks, match_signatures, reconstruct_errors


class PreservationChecks(unittest.TestCase):
    def test_permuted_endmembers_preserve_reconstruction_and_alignment(self):
        reference = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        truth = np.array([[.2, .3, .5], [.7, .1, .2]])
        order = np.array([2, 0, 1])
        spectra = truth @ reference
        estimated, abundance = reference[order], truth[:, order]
        permutation, sad = match_signatures(estimated, reference)
        np.testing.assert_allclose(abundance[:, permutation], truth)
        self.assertLess(float(sad.max()), .002)
        mse, _ = reconstruct_errors(spectra, abundance, estimated)
        np.testing.assert_allclose(mse, 0, atol=1e-30)

    def test_perfect_reconstruction_can_hide_wrong_signatures_and_abundances(self):
        reference = np.eye(2)
        truth = np.array([[.2, .8], [.4, .6]])
        estimated = np.array([[.5, .5], [0., 1.]])
        abundance = np.array([[.4, .6], [.8, .2]])
        spectra = truth @ reference
        mse, _ = reconstruct_errors(spectra, abundance, estimated)
        permutation, sad = match_signatures(estimated, reference)
        np.testing.assert_allclose(mse, 0, atol=1e-30)
        self.assertGreater(float(sad.mean()), 20)
        self.assertGreater(float(np.sqrt(np.mean((abundance[:, permutation]-truth)**2))), .2)

    def test_reference_masks_and_disjoint_error_decomposition(self):
        truth = np.array([[1., 0.], [.5, .5], [0., 1.], [0., 1.], [0., 1.], [0., 1.]])
        spectra = np.arange(12, dtype=float).reshape(6, 2) / 12
        masks, _, _ = make_masks(truth, spectra, (2, 3), ['Soil', 'Water'])
        self.assertEqual(int(masks['mixed_t080'].sum()), 1)
        self.assertEqual(int(masks['material_soil'].sum()), 2)
        self.assertFalse(np.any(masks['critical'] & masks['noncritical']))
        self.assertTrue(np.all(masks['critical'] | masks['noncritical']))
        error = np.array([.01, .04, .09, .16, .25, .36])
        contributions = [error[mask.ravel()].sum() for mask in (masks['critical'], masks['noncritical'])]
        self.assertAlmostEqual(sum(contributions), error.sum())


if __name__ == '__main__':
    unittest.main()
