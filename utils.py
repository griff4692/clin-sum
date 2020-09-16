import unicodedata


def decode_utf8(str):
    """
    :param str: string with utf-8 characters
    :return: string with all ascii characters

    This is necessary for the ROUGE nlp package consumption.
    """
    return unicodedata.normalize(u'NFKD', str).encode('ascii', 'ignore').decode('utf8').strip()


def tens_to_np(t):
    try:
        return t.numpy()
    except:
        return t.cpu().numpy()