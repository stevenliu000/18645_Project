inline int index_transform(int index, int num_col, int pad_size) {
    int new_index;
    if (index < pad_size)
        new_index = pad_size - index;
    else if (index >= num_col + pad_size)
        new_index = num_col - 2 - (index - num_col - pad_size);
    else
        new_index = index - pad_size;
    
    return new_index;
}