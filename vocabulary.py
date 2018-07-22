def create_vocabulary(dataset, terminal_token='<e>'):
    """Map each token in the dataset to a unique integer id.
    :param dataset a 2-d array, contains sequences of tokens.
                    A token can be a word or a character, depending on the problem.
    :param terminal_token (Optional). If specified, will be added to the vocabulary with id=0.
    :returns a tuple of vocabulary (a map from token to unique id) and reversed
        vocabulary (a map from unique ids to tokens).
    """
    vocabulary = {}
    reverse_vocabulary = []
    if terminal_token is not None:
        vocabulary[terminal_token] = 0
        reverse_vocabulary.append(terminal_token)
    for sample in dataset:
        for token in sample:
            if not token in vocabulary:
                vocabulary[token] = len(vocabulary)
                reverse_vocabulary.append(token)
    return vocabulary, reverse_vocabulary


def convert_to_ids(dataset, vocabulary):
    """Convert tokens to integers.
    :param dataset a 2-d array, contains sequences of tokens
    :param vocabulary a map from tokens to unique ids
    :returns a 2-d arrays, contains sequences of unique ids (integers)
    """
    return [[vocabulary[token] for token in sample] for sample in dataset]


def append_terminal_token(dataset, terminal_token='<e>'):
    """Appends end-of-sequence token to each sequence.
    :param dataset - a 2-d array, contains sequences of tokens.
    :param terminal_token (Optional) which symbol to append.
    """
    return [sample + [terminal_token] for sample in dataset]


def unify_seq_len(dataset, seq_len, default_id=0):
    """Cut or extend each sequence in dataset to have the same length.
    :param dataset a 2-d array, contains sequences of token ids.
    :param seq_len integer, desired sequence length.
    :param default_id if a sequence is shorter than seq_len, default_id
            will be appended to the end.
    :return a 2-d array, contains sequences of token ids, each sequence
            has length seq_len
    """
    return [sample[:seq_len] if seq_len < len(sample) else
         sample + [default_id for _ in range(0, seq_len - len(sample))]
         for sample in dataset]


def convert_to_tokens(sequence, reverse_vocabulary, terminal_id=0):
    """Convert a token id sequence back to tokens
    :param sequence a 1-d array of token ids.
    :param reverse_vocabulary a map from token ids to tokens.
    :param terminal_id a token id indicating end of sequence.
    :returns a sequence of tokens that does not include any tokens after terminal_id."""
    tokens = []
    for token_id in sequence:
        if token_id == terminal_id:
            break
        tokens.append(reverse_vocabulary[token_id])
    return tokens
