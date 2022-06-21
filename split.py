import random
from typing import List, Any, Optional, Tuple


def _atomic_split(index_: List[int],
                  splits: List[Tuple[Any, float]],
                  shuffle: bool = True,
                  random_seed: int = 42) -> List[Tuple[int, Any]]:
    """The atomic split function"""

    random.seed(random_seed)

    index_ = index_.copy()

    if shuffle:
        random.shuffle(index_)

    results = []

    if len(index_) > len(splits):
        sample_sizes = [int(s[1] * len(index_)) for s in splits]  # Compute each split's sample size

        # Rebalance samples size to have at least one element per sample if possible,
        # substracting added element from a random sample than has more than one.
        while 0 in sample_sizes and any([sz > 1 for sz in sample_sizes]):
            zero_index = sample_sizes.index(0)
            stock = random.choice([sz for sz in sample_sizes if sz > 1])
            stock_index = sample_sizes.index(stock)
            sample_sizes[zero_index] += 1
            sample_sizes[stock_index] -= 1

        for s, k in zip(splits, sample_sizes):  # len(splits) = len(sample_sizes)
            # try:
            results += [(index_.pop(0), s[0]) for _ in range(k)]  # Pop the first index k times
            # except IndexError:
            #     print(sample_sizes)

    if len(index_) <= len(splits):  # If there are any left or if very small group, distribute with priority
        results += [(index_.pop(0), splits[i][0]) for i in range(len(index_))]

    assert not index_

    return results


def _sort_output(output: List[Tuple[int, Any]]):
    return [el[1] for el in sorted(output, key=lambda x: x[0])]


def _print_stats(output: List[Any], splits: List[Tuple[Any, float]]):
    for s in splits:
        count = output.count(s[0])
        print(f"""Split {s[0]} got {count} examples (effective ratio: {count / len(output)}, expected: {s[1]})""")


def split(data_length: int,
          splits: List[Tuple[Any, float]],
          strats: Optional[List[Any]] = None,
          shuffle: bool = True,
          random_seed: int = 42,
          ) -> List[Any]:
    """Creates a split index for data of length `data_length` and according to the desired `splits`.

    This function returns a list of length `data_length` representing the distribution of splits, for instance,
    `['train', 'dev', 'dev', ... , 'test']`. On the contrary

    Args:
        data_length: The length of the data. For instance, if your data is a `pandas.DataFrame`, then you should set it
                     to `len(df)`.
        splits: A list of tuples, where each tuple specifies the name/number/id of split and its ratio. For instance,
                `[('train', 0.65), ('dev1', 0.10), ('dev2', 0.10), ('test', 0.15)]`.
                Note:
                    - Split-ratios should sum to 1.
                    - You can prioritise the distribution by ordering this list. For very tiny datasets, you sometimes
                      want to make sure that e.g. your test set gets available examples in priority. In that case,
                      you just want to set your test tuple as the first of the list (etc..)
        strats: A list of length `data_length` to be passed split data in a stratified fashion. If provided, splitting
                is done at the level of each subset.
        shuffle: If set to false, data gets distributed into splits in a linear manner.
        random_seed: For reproducibility.

    Returns:
         A list of length `data_length` representing the distribution of splits.
    """

    assert sum([s[1] for s in splits]) == 1.0, """`splits` ratios should sum to 1"""



    index = list(range(data_length))

    if not strats:
        output = _sort_output(_atomic_split(index, splits=splits))
        _print_stats(output, splits)
        return output

    else:
        assert len(strats) == data_length, """`strats` must be a list of length `data_length`"""

        outputs = []

        for strat in set(strats):
            strat_index = [idx for idx, strat_ in zip(index, strats) if strat_ == strat]
            outputs += _atomic_split(strat_index, splits, shuffle=shuffle, random_seed=random_seed)

        outputs = _sort_output(outputs)
        _print_stats(outputs, splits)
        return outputs


# Todo : add version
# Todo: add no dependency needed


# %%
from ajmc.commons.miscellaneous import read_google_sheet

sheet_id = '1_hDP_bGDNuqTPreinGS9-ShnXuXCjDaEbz-qEMUSito'
df = read_google_sheet(sheet_id, 'olr_gt')

def get_coarse_type(x):
    coarse_type_mapping = {
        # 'commentary+translation': 'commentary',
        # 'commentary+primary': 'commentary',
        'addenda': 'paratext',
        'appendix': 'paratext',
        'bibliography': 'structured_text',
        # 'hypothesis': 'paratext',
        'index': 'structured_text',
        'introduction': 'paratext',
        'preface': 'paratext',
        'title': 'structured_text',
        'toc': 'structured_text',
        'translation': 'paratext'}

    return coarse_type_mapping.get(x, x)

df['coarse_layout_type'] = df['layout_type'].apply(get_coarse_type)

a = '\n'.join(df['coarse_layout_type'].tolist())
#%%

stratify = [id + layout_type for id, layout_type in zip(df['commentary_id'], df['coarse_layout_type'])]
splits = [('test', 0.125),
          ('train', 0.75),
          ('dev', 0.125)]

split_col = split(data_length=len(df.index), splits=splits, strats=stratify)

a = '\n'.join(split_col)


#%%

from ajmc.commons.file_management.spreadsheets import check_entire_via_spreadsheets_conformity
