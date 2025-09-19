from datasets import load_dataset

def load_humaneval(run_sample=True, n=None, sample_frac=None, shuffle=True, seed=42):
    ds = load_dataset("openai_humaneval")["test"]
    if run_sample:
        if sample_frac is not None:  # use fraction
            n = max(1, int(sample_frac * len(ds)))
        if n is None:
            n = 10  # default if nothing given
        ds = ds.shuffle(seed=seed) if shuffle else ds
        ds = ds.select(range(min(n, len(ds))))
    return ds