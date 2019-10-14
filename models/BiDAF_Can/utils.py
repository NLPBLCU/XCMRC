def save_params(file_name, params, **kwargs):
    """
        Save params to file_name.
        params: a list of Theano variables
        """
    dic = {'params': [x.get_value() for x in params]}
    dic.update(kwargs)
    with gzip.open(file_name, "w") as save_file:
        pickle.dump(obj=dic, file=save_file, protocol=-1)


def load_params(file_name):
    """
        Load params from file_name.
        """
    with gzip.open(file_name, "rb") as save_file:
        dic = pickle.load(save_file)
    return dic
