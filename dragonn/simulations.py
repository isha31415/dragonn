from __future__ import absolute_import, division, print_function
import numpy as np
import pdb
import random
from dragonn.synthetic.synthetic import (
    RepeatedEmbedder, SubstringEmbedder, ReverseComplementWrapper,
    UniformPositionGenerator, InsideCentralBp,
    LoadedEncodeMotifs, PwmSamplerFromLoadedMotifs,
    UniformIntegerGenerator, ZeroOrderBackgroundGenerator,
    EmbedInABackground, GenerateSequenceNTimes,
    RandomSubsetOfEmbedders, IsInTraceLabelGenerator,
    EmbeddableEmbedder, PairEmbeddableGenerator,
)
from dragonn.synthetic.util import DiscreteDistribution
from utils import one_hot_encode
from pkg_resources import resource_filename
from models import SequenceDNN
from models import Model
import matplotlib.pyplot as plt


ENCODE_MOTIFS_PATH = resource_filename('dragonn.synthetic', 'motifs.txt.gz')
loaded_motifs = LoadedEncodeMotifs(ENCODE_MOTIFS_PATH, pseudocountProb=0.001)

deep_cnn_MOTIF_DENSITY_parameters = {
    'seq_length': 500,
    #'use_deep_CNN': True, # we have to specify this option when using a deep CNN
    'num_filters': (15,),
    'conv_width': (15,)}

deep_cnn_SIMPLE_MOTIF_EMBEDDING_parameters = {
    'seq_length': 500,
    #'use_deep_CNN': True, # we have to specify this option when using a deep CNN
    'num_filters': (15,),
    'conv_width': (15,)}

deep_cnn_SIMULATE_SINGLE_MOTIF_DETECTION_parameters = {
    'seq_length': 500,
    #'use_deep_CNN': True, # we have to specify this option when using a deep CNN
    'num_filters': (15,),
    'conv_width': (15,)}

deep_cnn_SIMULATE_MOTIF_COUNTING_parameters = {
    'seq_length': 500,
    #'use_deep_CNN': True, # we have to specify this option when using a deep CNN
    'num_filters': (15,),
    'conv_width': (15,)}

deep_cnn_SIMULATE_MOTIF_DENSITY_LOCALIZATION_parameters = {
    'seq_length': 500,
    #'use_deep_CNN': True, # we have to specify this option when using a deep CNN
    'num_filters': (15,),
    'conv_width': (15,)}

deep_cnn_SIMULATE_MULTI_MOTIF_EMBEDDING_parameters = {
    'seq_length': 500,
    #'use_deep_CNN': True, # we have to specify this option when using a deep CNN
    'num_filters': (15,),
    'conv_width': (15,)}

deep_cnn_SIMULATE_DIFFERENTIAL_ACCESSIBILITY_parameters = {
    'seq_length': 500,
    #'use_deep_CNN': True, # we have to specify this option when using a deep CNN
    'num_filters': (15,),
    'conv_width': (15,)}

deep_cnn_SIMULATE_HETERODIMER_GRAMMAR_parameters = {
    'seq_length': 500,
    #'use_deep_CNN': True, # we have to specify this option when using a deep CNN
    'num_filters': (15,),
    'conv_width': (15,)}

def Deep_Lift_scores(motif_name, seq_length, num_seq,
                  min_counts, max_counts, GC_fraction,
                  central_bp=None):

    pos_neg(motif_name, seq_length, num_seq,
                  min_counts, max_counts, GC_fraction,
                  central_bp=None)

    train(self, training_set, training_labels, validation_set)

def get_distribution(GC_fraction):
    return DiscreteDistribution({
        'A': (1 - GC_fraction) / 2, 'C': GC_fraction / 2,
        'G': GC_fraction / 2, 'T': (1 - GC_fraction) / 2})

def simple_motif_embedding(motif_name, seq_length, num_seq, GC_fraction):
    """
    returns sequence array
    """
    if motif_name is None:
        embedders = []
    else:
        substring_generator = PwmSamplerFromLoadedMotifs(
            loaded_motifs, motif_name)
        embedders = [SubstringEmbedder(
            ReverseComplementWrapper(substring_generator))]
    embed_in_background = EmbedInABackground(
        ZeroOrderBackgroundGenerator(
            seq_length, discreteDistribution=get_distribution(GC_fraction)),
        embedders)
    generated_sequences = GenerateSequenceNTimes(
        embed_in_background, num_seq).generateSequences()
    sequence__and_embedding_arr = np.asarray(
        [[generated_seq.seq, generated_seq.embeddings]
         for generated_seq in generated_sequences])
    sequence_arr = sequence__and_embedding_arr[:, 0]
    embedding_arr = sequence__and_embedding_arr[:, 1]
    #embeddings_for_each_seq = [generated_seq.embeddings for generated_seq in generated_sequences]
    return sequence_arr, embedding_arr
    #return sequence_arr, embeddings_for_each_seq

def motif_density(motif_name, seq_length, num_seq,
                  min_counts, max_counts, GC_fraction,
                  central_bp=None):
    """
    returns sequences with motif density.
    """

    substring_generator = PwmSamplerFromLoadedMotifs(loaded_motifs, motif_name)
    if central_bp is not None:
        position_generator = InsideCentralBp(central_bp)
    else:
        position_generator = UniformPositionGenerator()
    quantity_generator = UniformIntegerGenerator(min_counts, max_counts)
    embedders = [
        RepeatedEmbedder(
            SubstringEmbedder(
                ReverseComplementWrapper(
                    substring_generator), position_generator),
            quantity_generator)]
    embed_in_background = EmbedInABackground(
        ZeroOrderBackgroundGenerator(
            seq_length, discreteDistribution=get_distribution(GC_fraction)),
        embedders)
    generated_sequences = GenerateSequenceNTimes(
        embed_in_background, num_seq).generateSequences()
    sequence__and_embedding_arr = np.asarray(
        [[generated_seq.seq, generated_seq.embeddings]
         for generated_seq in generated_sequences])
    sequence_arr = sequence__and_embedding_arr[:, 0]
    embedding_arr = sequence__and_embedding_arr[:, 1]
    positions_array=[]
    motif_name_array=[]
    #embeddings_for_each_seq = [generated_seq.embeddings for generated_seq in generated_sequences]
    for x in range(0,len(sequence__and_embedding_arr)):
        for y in range(0, len(sequence__and_embedding_arr[x])):
            #try:
            #positions_array.append(embedding_arr[x][y].what)
            motif_name_array.append(embedding_arr[x][y].startPos)
            #except:
            #    print("Warning! cannot call 'what' on a string object, skipping this embedding")
    return sequence_arr, embedding_arr,positions_array,motif_name_array
    #return sequence_arr, embeddings_for_each_seq

def pos_neg(motif_name, seq_length, num_seq,
                  min_counts, max_counts, GC_fraction,
                  central_bp=None):

    #positive results test
    positive_set,positive_embedding,positive_positions_arr,positive_motif_name_arr = motif_density(motif_name, seq_length, num_seq,
                      min_counts, max_counts, GC_fraction,
                      central_bp=None)
    #pdb.set_trace()

    random.shuffle(positive_set)
    thresh_positive=int(0.3*(len(positive_set)))
    validation_positive_set = positive_set[0:thresh_positive]
    training_positive_set = positive_set[thresh_positive:]

    negative_set,negative_embedding,negative_positions_arr,negative_motif_name_arr = motif_density(motif_name, seq_length, num_seq,
                      min_counts = 0, max_counts = 0, GC_fraction = .4,
                      central_bp=None)
    random.shuffle(negative_set)
    thresh_negative=int(0.3*(len(negative_set)))
    validation_negative_set = negative_set[0:thresh_negative]
    training_negative_set = negative_set[thresh_negative:]

    validation_set = np.concatenate((validation_negative_set, validation_positive_set), axis = 0)
    training_set = np.concatenate((training_negative_set, training_positive_set), axis = 0)


    positive_labels = np.ones(training_positive_set.shape)
    positive_labels=np.reshape(positive_labels,(len(positive_labels),1))
    negative_labels = np.zeros(training_negative_set.shape)
    negative_labels=np.reshape(negative_labels,(len(negative_labels),1))
    training_labels = np.concatenate((positive_labels, negative_labels),axis=0)

    pos_val_labels=np.ones(validation_positive_set.shape)
    pos_val_labels=np.reshape(pos_val_labels,(len(pos_val_labels),1))
    neg_val_labels=np.zeros(validation_negative_set.shape)
    neg_val_labels=np.reshape(neg_val_labels,(len(neg_val_labels),1))
    validation_labels=np.concatenate((pos_val_labels,neg_val_labels),axis=0)
    #pdb.set_trace()
    training_set=one_hot_encode(np.array([i for i in training_set]));
    validation_set=one_hot_encode(np.array([i for i in validation_set]));
    return training_labels,training_set,validation_labels,validation_set,positive_positions_arr,positive_motif_name_arr,negative_positions_arr,negative_motif_name_arr

def simulate_single_motif_detection(motif_name, seq_length,
                                    num_pos, num_neg, GC_fraction):
    """
    Simulates two classes of seqeuences:
        - Positive class sequence with a motif
          embedded anywhere in the sequence
        - Negative class sequence without the motif

    Parameters
    ----------
    motif_name : str
        encode motif name
    seq_length : int
        length of sequence
    num_pos : int
        number of positive class sequences
    num_neg : int
        number of negative class sequences
    GC_fraction : float
        GC fraction in background sequence

    Returns
    -------
    sequence_arr : 1darray
        Array with sequence strings.
    y : 1darray
        Array with positive/negative class labels.
    """
    motif_sequence_arr = simple_motif_embedding(
        motif_name, seq_length, num_pos, GC_fraction, embedding_arr)
    random_sequence_arr = simple_motif_embedding(
        None, seq_length, num_neg, GC_fraction, embedding_arr)
    sequence_arr = np.concatenate((motif_sequence_arr, random_sequence_arr))
    y = np.array([[True]] * num_pos + [[False]] * num_neg)
    sequence__and_embedding_arr = np.asarray(
        [[generated_seq.seq, generated_seq.embeddings]
         for generated_seq in generated_sequences])
    sequence_arr = sequence__and_embedding_arr[:, 0]
    embedding_arr = sequence__and_embedding_arr[:, 1]
    #embeddings_for_each_seq = [generated_seq.embeddings for generated_seq in generated_sequences]
    return sequence_arr, embedding_arr
    #return sequence_arr, embeddings_for_each_seq

def simulate_motif_counting(motif_name, seq_length, pos_counts, neg_counts,
                            num_pos, num_neg, GC_fraction):
    """
    Generates data for motif counting task.
    Parameters
    ----------
    motif_name : str
    seq_length : int
    pos_counts : list
        (min_counts, max_counts) for positive set.
    neg_counts : list
        (min_counts, max_counts) for negative set.
    num_pos : int
    num_neg : int
    GC_fraction : float
    Returns
    -------
    sequence_arr : 1darray
        Contains sequence strings.
    y : 1darray
        Contains labels.
    """
    pos_count_sequence_array = motif_density(
        motif_name, seq_length, num_pos,
        pos_counts[0], pos_counts[1], GC_fraction, embedding_arr)
    neg_count_sequence_array = motif_density(
        motif_name, seq_length, num_pos,
        neg_counts[0], neg_counts[1], GC_fraction, embedding_arr)
    sequence_arr = np.concatenate(
        (pos_count_sequence_array, neg_count_sequence_array, embedding_arr))
    y = np.array([[True]] * num_pos + [[False]] * num_neg)
    sequence__and_embedding_arr = np.asarray(
        [[generated_seq.seq, generated_seq.embeddings]
         for generated_seq in generated_sequences])
    sequence_arr = sequence__and_embedding_arr[:, 0]
    embedding_arr = sequence__and_embedding_arr[:, 1]
    #embeddings_for_each_seq = [generated_seq.embeddings for generated_seq in generated_sequences]
    return sequence_arr, embedding_arr
    #return sequence_arr, embeddings_for_each_seq

def simulate_motif_density_localization(
        motif_name, seq_length, center_size, min_motif_counts,
        max_motif_counts, num_pos, num_neg, GC_fraction):
    """
    Simulates two classes of seqeuences:
        - Positive class sequences with multiple motif instances
          in center of the sequence.
        - Negative class sequences with multiple motif instances
          anywhere in the sequence.
    The number of motif instances is uniformly sampled
    between minimum and maximum motif counts.

    Parameters
    ----------
    motif_name : str
        encode motif name
    seq_length : int
        length of sequence
    center_size : int
        length of central part of the sequence where motifs can be positioned
    min_motif_counts : int
        minimum number of motif instances
    max_motif_counts : int
        maximum number of motif instances
    num_pos : int
        number of positive class sequences
    num_neg : int
        number of negative class sequences
    GC_fraction : float
        GC fraction in background sequence

    Returns
    -------
    sequence_arr : 1darray
        Contains sequence strings.
    y : 1darray
        Contains labels.
    """
    localized_density_sequence_array = motif_density(
        motif_name, seq_length, num_pos,
        min_motif_counts, max_motif_counts, GC_fraction, center_size, embedding_arr)
    unlocalized_density_sequence_array = motif_density(
        motif_name, seq_length, num_neg,
        min_motif_counts, max_motif_counts, GC_fraction, embedding_arr)
    sequence_arr = np.concatenate(
        (localized_density_sequence_array, unlocalized_density_sequence_array, embedding_arr))
    y = np.array([[True]] * num_pos + [[False]] * num_neg)
    sequence__and_embedding_arr = np.asarray(
        [[generated_seq.seq, generated_seq.embeddings]
         for generated_seq in generated_sequences])
    sequence_arr = sequence__and_embedding_arr[:, 0]
    embedding_arr = sequence__and_embedding_arr[:, 1]
    #embeddings_for_each_seq = [generated_seq.embeddings for generated_seq in generated_sequences]
    return sequence_arr, embedding_arr
    #return sequence_arr, embeddings_for_each_seq

def simulate_multi_motif_embedding(motif_names, seq_length, min_num_motifs,
                                   max_num_motifs, num_seq, GC_fraction):
    """
    Generates data for multi motif recognition task.
    Parameters
    ----------
    motif_names : list
        List of strings.
    seq_length : int
    min_num_motifs : int
    max_num_motifs : int
    num_seq : int
    GC_fraction : float
    Returns
    -------
    sequence_arr : 1darray
        Contains sequence strings.
    y : ndarray
        Contains labels for each motif.
    """
    print("hi")
    def get_embedder(motif_name):
        substring_generator = PwmSamplerFromLoadedMotifs(
            loaded_motifs, motif_name)
        return SubstringEmbedder(
                ReverseComplementWrapper(substring_generator),
                name=motif_name)

    embedders = [get_embedder(motif_name) for motif_name in motif_names]
    quantity_generator = UniformIntegerGenerator(
        min_num_motifs, max_num_motifs)
    combined_embedder = [RandomSubsetOfEmbedders(
        quantity_generator, embedders)]
    embed_in_background = EmbedInABackground(
        ZeroOrderBackgroundGenerator(
            seq_length, discreteDistribution=get_distribution(GC_fraction)),
        combined_embedder)
    generated_sequences = GenerateSequenceNTimes(
        embed_in_background, num_seq).generateSequences()
    label_generator = IsInTraceLabelGenerator(np.asarray(motif_names))
    data_arr = np.asarray(
        [[generated_seq.seq] + label_generator.generateLabels(generated_seq)
         for generated_seq in generated_sequences])
    sequence_arr = data_arr[:, 0]
    y = data_arr[:, 1:].astype(bool)
    sequence__and_embedding_arr = np.asarray(
        [[generated_seq.seq, generated_seq.embeddings]
         for generated_seq in generated_sequences])
    sequence_arr = sequence__and_embedding_arr[:, 0]
    embedding_arr = sequence__and_embedding_arr[:, 1]
    #embeddings_for_each_seq = [generated_seq.embeddings for generated_seq in generated_sequences]
    return sequence_arr, embedding_arr
    #return sequence_arr, embeddings_for_each_seq

def simulate_differential_accessibility(
        pos_motif_names, neg_motif_names, seq_length,
        min_num_motifs, max_num_motifs, num_pos, num_neg, GC_fraction):
    """
    Generates data for differential accessibility task.

    Parameters
    ----------
    pos_motif_names : list
        List of strings.
    neg_motif_names : list
        List of strings.
    seq_length : int
    min_num_motifs : int
    max_num_motifs : int
    num_pos : int
    num_neg : int
    GC_fraction : float

    Returns
    -------
    sequence_arr : 1darray
        Contains sequence strings.
    y : 1darray
        Contains labels.
    """
    pos_motif_sequence_arr, _ = simulate_multi_motif_embedding(
        pos_motif_names, seq_length,
        min_num_motifs, max_num_motifs, num_pos, GC_fraction)
    neg_motif_sequence_arr, _ = simulate_multi_motif_embedding(
        neg_motif_names, seq_length,
        min_num_motifs, max_num_motifs, num_neg, GC_fraction)
    sequence_arr = np.concatenate(
        (pos_motif_sequence_arr, neg_motif_sequence_arr))
    y = np.array([[True]] * num_pos + [[False]] * num_neg)
    #BELOW IS ADDED
    sequence__and_embedding_arr = np.asarray(
        [[generated_seq.seq, generated_seq.embeddings]
         for generated_seq in generated_sequences])
    sequence_arr = sequence__and_embedding_arr[:, 0]
    embedding_arr = sequence__and_embedding_arr[:, 1]
    #embeddings_for_each_seq = [generated_seq.embeddings for generated_seq in generated_sequences]
    return sequence_arr, embedding_arr
    #return sequence_arr, embeddings_for_each_seq

def simulate_heterodimer_grammar(
        motif1, motif2, seq_length,
        min_spacing, max_spacing, num_pos, num_neg, GC_fraction):
    """
    Simulates two classes of sequences with motif1 and motif2:
        - Positive class sequences with motif1 and motif2 positioned
          min_spacing and max_spacing
        - Negative class sequences with independent motif1 and motif2 positioned
        anywhere in the sequence, not as a heterodimer grammar

    Parameters
    ----------
    seq_length : int, length of sequence
    GC_fraction : float, GC fraction in background sequence
    num_pos : int, number of positive class sequences
    num_neg : int, number of negatice class sequences
    motif1 : str, encode motif name
    motif2 : str, encode motif name
    min_spacing : int, minimum inter motif spacing
    max_spacing : int, maximum inter motif spacing

    Returns
    -------
    sequence_arr : 1darray
        Array with sequence strings.
    y : 1darray
        Array with positive/negative class labels.
    """

    motif1_generator = PwmSamplerFromLoadedMotifs(loaded_motifs, motif1)
    motif2_generator = PwmSamplerFromLoadedMotifs(loaded_motifs, motif2)
    separation_generator = UniformIntegerGenerator(min_spacing, max_spacing)
    embedder = EmbeddableEmbedder(PairEmbeddableGenerator(
        motif1_generator, motif2_generator, separation_generator))
    embed_in_background = EmbedInABackground(ZeroOrderBackgroundGenerator(
        seq_length, discreteDistribution=get_distribution(GC_fraction)), [embedder])
    generated_sequences = GenerateSequenceNTimes(
        embed_in_background, num_pos).generateSequences()
    grammar_sequence_arr = np.asarray(
        [generated_seq.seq for generated_seq in generated_sequences])
    nongrammar_sequence_arr, _ = simulate_multi_motif_embedding(
        [motif1, motif2], seq_length, 2, 2, num_neg, GC_fraction)
    sequence_arr = np.concatenate(
        (grammar_sequence_arr, nongrammar_sequence_arr))
    y = np.array([[True]] * num_pos + [[False]] * num_neg)
    sequence__and_embedding_arr = np.asarray(
        [[generated_seq.seq, generated_seq.embeddings]
         for generated_seq in generated_sequences])
    sequence_arr = sequence__and_embedding_arr[:, 0]
    embedding_arr = sequence__and_embedding_arr[:, 1]
    #embeddings_for_each_seq = [generated_seq.embeddings for generated_seq in generated_sequences]
    return sequence_arr, embedding_arr
    #return sequence_arr, embeddings_for_each_seq

############################################################################################################
def main(simulation_Name):
    print('There are eleven cats in the house')
    #sequences, embeddings = motif_density("TAL1_known4", 1000, 50, 2, 4, .4, central_bp=None)
    seqLen=500
    training_labels,training_set,validation_labels,validation_set,positive_pos,positive_motifs,negative_pos,negative_motifs=pos_neg("TAL1_known4",seqLen, 5000, 2, 4, .4, central_bp=None)
    print(len(training_set))
    if (simulation_Name == "Simple Motif Embedding"):
        myModel=SequenceDNN(**deep_cnn_SIMPLE_MOTIF_EMBEDDING_parameters)
    if (simulation_Name == "Motif Density"):
        myModel=SequenceDNN(**deep_cnn_MOTIF_DENSITY_parameters)
    if (simulation_Name == "Single Motif Detection"):
        myModel=SequenceDNN(**deep_cnn_SIMULATE_SINGLE_MOTIF_DETECTION_parameters)
    if (simulation_Name == "Motif Counting"):
        myModel=SequenceDNN(**deep_cnn_SIMULATE_MOTIF_COUNTING_parameters)
    if (simulation_Name == "Motif Density Localization"):
        myModel=SequenceDNN(**deep_cnn_SIMULATE_MOTIF_DENSITY_LOCALIZATION_parameters)
    if (simulation_Name == "Multi Motif Embedding"):
        myModel=SequenceDNN(**deep_cnn_SIMULATE_MULTI_MOTIF_EMBEDDING_parameters)
    if (simulation_Name == "Differential Accessibility"):
        myModel=SequenceDNN(**deep_cnn_SIMULATE_DIFFERENTIAL_ACCESSIBILITY_parameters)
    if (simulation_Name == "Heterodimer Grammar"):
        myModel=SequenceDNN(**deep_cnn_SIMULATE_HETERODIMER_GRAMMAR_parameters)
    training_labels=np.array([int(i) for i in training_labels])
    validation_labels=np.array([int(i) for i in validation_labels])
    training_labels=np.reshape(training_labels,(len(training_labels),1))
    validation_labels=np.reshape(validation_labels,(len(validation_labels),1))
    myModel.train(training_set,training_labels.astype(np.bool),tuple([validation_set,validation_labels.astype(np.bool)]))
    dLArray=myModel.deeplift(training_set)
    print(dLArray)

if __name__=="__main__":
    main("Heterodimer Grammar")

'''
def histogram_positive_motifs():
    a = [this is our array] #positive deeplift scores
    plt.hist(a, bins = 100, normed = False, density = False)
    plt.title('Instances of Positive Motifs')
    plt.show()

def histogram_negative_motifs():
    a = [this is our array] #negative deeplift scores
    plt.hist(a, bins = 100, normed = False, density = False)
    plt.title('Instances of Negative Motif Scores')
    plt.show()
'''
