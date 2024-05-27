import pandas as pd
import numpy as np
import networkx as nx

from graspologic.embed import node2vec_embed
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

def getNodeEmbeddings(input_file, output_file):
    """
    Computes the node embeddings (dim = 128) of each node in a given network.

    Args:
        input_file: path of the graph file in .gexf extension
        output_file: path where to save the output file
    Returns:

    """
    G = nx.read_gexf(input_file)
    G = nx.Graph(G)

    node2vec_embedding, node_labels = node2vec_embed(
        G, num_walks=16, walk_length=16, workers=16, inout_hyperparameter=1.0, return_hyperparameter=1.0
    )
    
    np.save(output_file+'_embed.npy', node2vec_embedding) # saves the emebddings
    np.save(output_file+'_nodeLabels.npy', node_labels) #saves the node labels relative to the embeddings files

def classify_embeddings(node2vec_embedding, node_labels, label_map):
    """
    Trains a random forest classifier on node embeddings, using 10 Fold Cross-Validation

    Args:
        node2vec_embedding:numpy array of node embeddings
        node_labels: numpy array of node labels
        label_map: dataframe with the classification label for each userid (columns = ['userid', 'label'])
    Returns:
        model: trained model
        n_scores: 10-fold cross-validation performance
    """
    node_labels = node_labels.astype(str)
    label_map = label_map[['userid', 'label']].set_index('userid').T.to_dict('list')
      
    node_colours = []
    
    for target in node_labels:
        if target in label_map.keys():
            node_colours.append(label_map[target][0])
        else: 
            node_colours.append(np.NaN)
    
    df = pd.DataFrame(node2vec_embedding)
    df['label'] = node_colours
    df.dropna(inplace=True)
    
    model = RandomForestClassifier(class_weight='balanced')
    cv = StratifiedKFold(n_splits=10)
    n_scores = cross_validate(model, df[[i for i in range(0, 128)]], df['label'], scoring=['accuracy', 'recall', 'precision','f1', 'roc_auc'], cv=cv, n_jobs=-1, error_score='raise')
    return n_scores