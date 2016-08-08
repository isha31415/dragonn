def deeplift(self, X, batch_size=200):
    """
    Returns (num_task, num_samples, 1, num_bases, sequence_length) deeplift score array.
    """
    assert len(np.shape(X)) == 4 and np.shape(X)[1] == 1
    # normalize sequence convolution weights
    kc.mean_normalise_first_conv_layer_weights(self.model, None)
    # run deeplift
    deeplift_model = kc.convert_sequential_model(
        self.model, mxts_mode=MxtsMode.DeepLIFT)
    target_contribs_func = deeplift_model.get_target_contribs_func(
        find_scores_layer_idx=0)
    return np.asarray([
        target_contribs_func(task_idx=i, input_data_list=[X],
                             batch_size=batch_size, progress_update=None)
        for i in range(self.num_tasks)])
