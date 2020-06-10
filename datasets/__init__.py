def get_dataset(split, hps):
    if hps.dataset == 'maf':
        from .maf import Dataset
        dataset = Dataset(hps.dfile, split, hps.batch_size)
    else:
        raise Exception()

    assert dataset.d == hps.dimension

    return dataset