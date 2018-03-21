

def check_dim(data, dim=4):
    """ Function to check data matrix dimensions

    :param data:
    :param dim:
    :return:
    """
    if len(data.shape) == dim:
        return True
    else:
        return False