from multi_news import load as multi_load


def load_dataset(name='multi_news'):
    if name == 'multi_news':
        return multi_load()
    else:
        raise Exception('Unrecognized dataset={}'.format(name))
