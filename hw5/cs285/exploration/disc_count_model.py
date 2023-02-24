import numpy as np
from torch import nn

from cs285.exploration.base_exploration_model import BaseExplorationModel


class DiscCountPointmassModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, disc_size=50, **kwargs):
        super().__init__(**kwargs)
        self.disc_factor = 1 / disc_size
        self.count_table = np.full((disc_size + 1, disc_size + 1), 1)

    def forward(self, ob_no):
        assert len(ob_no.shape) == 2

        idx0 = np.array(ob_no[:, 0] / self.disc_factor, dtype=int)
        idx1 = np.array(ob_no[:, 1] / self.disc_factor, int)
        counts = self.count_table[idx0, idx1]

        return 1 / counts

    def forward_np(self, ob_no):
        return self(ob_no)

    def update(self, ob_no):
        assert len(ob_no.shape) == 2

        idx0 = np.array(ob_no[:, 0] / self.disc_factor, dtype=int)
        idx1 = np.array(ob_no[:, 1] / self.disc_factor, int)

        self.count_table[idx0, idx1] += 1

        return {
            'Training Loss': 0,
        }


if __name__ == '__main__':
    model = DiscCountPointmassModel(None, None, disc_size=40)
    vec = 10
    s0 = np.array([0.4, 0.4])
    s00 = np.array([0.4, 0.42])
    s1 = np.array([0.4, 0.425])
    s2 = np.array([0.65, 0.4])

    s0v = np.repeat(s0[None], vec, axis=0)
    s00v = np.repeat(s00[None], vec, axis=0)
    s1v = np.repeat(s1[None], vec, axis=0)
    s2v = np.repeat(s2[None], vec, axis=0)

    ret1 = model(s0v)
    ret2 = model(s0v)
    ret3 = model(s00v)

    ret4 = model(s1v)
    ret5 = model(s2v)

    assert np.all(np.equal(ret1, ret2))
    assert np.all(np.equal(ret1, ret4))
    assert np.all(np.equal(ret1, ret5))
    assert np.all(np.equal(ret1, ret3))

    model.update(s0v)

    ret1 = model(np.array(s0v))
    ret2 = model(np.array(s0v))
    ret3 = model(np.array(s00v))

    ret4 = model(np.array(s1v))
    ret5 = model(np.array(s2v))

    assert np.all(np.equal(ret1, ret2))
    assert np.all(np.equal(ret1, ret3))
    assert np.all(ret1 < ret4)
    assert np.all(ret1 < ret5)

    model.update(s1v)
    model.update(s2v)

    ret1 = model(np.array(s0v))
    ret2 = model(np.array(s0v))
    ret3 = model(np.array(s00v))

    ret4 = model(np.array(s1v))
    ret5 = model(np.array(s2v))

    assert np.all(np.equal(ret1, ret2))
    assert np.all(np.equal(ret1, ret4))
    assert np.all(np.equal(ret1, ret5))
    assert np.all(np.equal(ret1, ret3))
