r"""Testing the Cartesian tessellation utilities implemented in
cards.slicer.cartesian_tessellation.
"""

# author: pthouvenin (pierre-antoine.thouvenin@centralelille.fr)

# adpated from: https://gitlab.cristal.univ-lille.fr/pthouven/dsgs

import numpy as np
import pytest

import cards.slicers.cartesian_tessellation as ct


@pytest.fixture
def N():
    return 9


@pytest.fixture
def nchunks():
    return 3


@pytest.fixture
def overlap():
    return 3


def test_local_split_range_single_chunk(N, nchunks, overlap):
    with pytest.raises(ValueError) as excinfo:
        ct.local_split_range(nchunks, N, 0, 4)
    assert "More than 100% overlap between two consecutive segments" in str(
        excinfo.value
    )


def test_fail_split_range_overlap(N, nchunks, overlap):
    with pytest.raises(ValueError) as excinfo:
        ct.split_range(nchunks, N, 4)
    assert "More than 100% overlap between two consecutive segments" in str(
        excinfo.value
    )


def test_fail_local_split_range_overlap(N, nchunks, overlap):
    with pytest.raises(ValueError) as excinfo:
        ct.local_split_range(nchunks, N, 0, 4)
    assert "More than 100% overlap between two consecutive segments" in str(
        excinfo.value
    )

    with pytest.raises(ValueError) as excinfo:
        ct.local_split_range(nchunks, N, N)
    assert "Index should be taken in [0, ..., nchunks-1], with nchunks={0}".format(
        nchunks
    ) in str(excinfo.value)


def test_split_range_no_overlap(N, nchunks, overlap):
    rg = ct.split_range(nchunks, N)
    # check that 2 consecutive start index are distant from 1
    assert np.allclose(rg[1:, 0] - rg[:-1, 1], 1)
    # check size of each chunk (same size for each in this case)
    assert np.allclose(np.diff(rg, n=1, axis=1) + 1, 3)

    # test single process and global versions coincide
    rg2 = np.concatenate(
        [ct.local_split_range(nchunks, N, k)[None, :] for k in range(nchunks)],
        axis=0,
    )
    assert np.allclose(rg, rg2)


def test_split_range_overlap(N, nchunks, overlap):
    rg = ct.split_range(nchunks, N, overlap)
    # check overlap between 2 consecutive segments (from the left)
    assert np.all(np.abs(rg[:-1, 1] - rg[1:, 0] + 1) == overlap)
    # test single process and global versions coincide
    rg2 = np.concatenate(
        [
            ct.local_split_range(nchunks, N, k, overlap=overlap)[None, :]
            for k in range(nchunks)
        ],
        axis=0,
    )
    assert np.allclose(rg, rg2)


def test_split_range_overlap_forward(N, nchunks, overlap):
    rg = ct.split_range(nchunks, N, overlap, False)
    # check overlap between 2 consecutive segments (from the left)
    assert np.allclose(np.abs(rg[:-1, 1] - rg[1:, 0] + 1), overlap)
    # test single process and global versions coincide
    rg2 = np.concatenate(
        [
            ct.local_split_range(
                nchunks,
                N,
                k,
                overlap=overlap,
                backward=False,
            )[None, :]
            for k in range(nchunks)
        ],
        axis=0,
    )
    assert np.allclose(rg, rg2)


def test_local_split_range(N, nchunks, overlap):
    rg = ct.split_range(nchunks, N, overlap)
    global_rg = np.concatenate(
        [
            ct.local_split_range(nchunks, N, k, overlap=overlap)[None, :]
            for k in range(nchunks)
        ],
        axis=0,
    )
    assert np.allclose(global_rg, rg)


def test_local_split_range_overlap_n(N, nchunks, overlap):
    rg = np.concatenate(
        (
            ct.local_split_range(nchunks, N, 1, overlap)[None, :],
            ct.local_split_range(nchunks, N, 0, overlap)[None, :],
        ),
        axis=0,
    )
    rg2 = ct.local_split_range_nd(
        np.array(2 * [nchunks]),
        np.array(2 * [N]),
        np.array([1, 0]),
        np.array(2 * [overlap]),
    )
    assert np.allclose(rg, rg2)


def test_split_range_interleaved_error(N):
    with pytest.raises(ValueError) as excinfo:
        ct.split_range_interleaved(N + 1, N)
    assert r"Number of segments nchunks={0} greater than the dimension N={1}".format(
        N + 1, N
    ) in str(excinfo.value)


def test_local_split_range_interleaved_error(N):
    with pytest.raises(ValueError) as excinfo:
        ct.local_split_range_interleaved(N + 1, N, 0)
    assert r"Number of segments nchunks={0} greater than the dimension N={1}".format(
        N + 1, N
    ) in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        ct.local_split_range_interleaved(2, N, N)
    assert r"Index should be taken in [0, ..., nchunks-1], with nchunks={0}".format(
        2
    ) in str(excinfo.value)


def test_split_range_interleaved(N, nchunks, overlap):
    rg = ct.split_range_interleaved(nchunks, N)
    assert np.all([rg[k].start == k for k in range(len(rg))])
    assert np.all([rg[k].stop == N for k in range(len(rg))])
    assert np.all([rg[k].step == nchunks for k in range(len(rg))])
    rg2 = ct.local_split_range_interleaved(nchunks, N, 0)
    assert np.all([rg2 == rg[0]])


def test_get_neighbour(N, nchunks, overlap):
    ranknd = np.array([0, 1], dtype="i")
    grid_size = np.array([nchunks, nchunks], dtype="i")
    disp = np.ones((2,), dtype="i")
    rank = ct.get_neighbour(ranknd, grid_size, disp)
    assert rank == (ranknd[0] + disp[0]) * grid_size[-1] + ranknd[-1] + disp[-1]  # = 5
