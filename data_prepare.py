#!/usr/bin/env python
# coding: utf-8

# In[26]:


import os
import pandas as pd
import numpy as np
import networkx
import matplotlib.pyplot as plt


# ##check work dir##

# In[2]:


cwd = os.getcwd()
print(cwd)


# In[3]:


#load data and labels
def load_data(dataset):
    path = "../../data/"+str(dataset)+"/"
    data = pd.read_csv(path+"abundance.tsv",header=None,sep="\t",index_col=0)
    with open(path+"labels.txt") as f:
        string = f.read().rstrip("\n")
        labels = string.split("\n")
    return data,labels


# In[4]:


# Input: features_raw(the species names) Output:a DataFrame index-species/genus columns-taxa level
def get_feature_df(features_raw):
    kingdom, phylum, cl, order, family, genus, species  = [], [], [], [], [], [], []
    for f in features_raw:

        name = f.split("k__")[1].split("|p__")[0].replace(".","")
        if "_unclassified" in name:
            name = 'unclassified_' + name.split("_unclassified")[0]
        kingdom.append(name)

        if "p__" in f:
            name =f.split("p__")[1].split("|c__")[0].replace(".","")
            if "_unclassified" in name:
                name = 'unclassified_' + name.split("_unclassified")[0]
            if name != "":
                phylum.append(name)
            else:
                phylum.append("NA")
        else:
            phylum.append("NA")

        if "c__" in f:
            name = f.split("c__")[1].split("|o__")[0].replace(".","")
            if "_unclassified" in name:
                name = 'unclassified_' + name.split("_unclassified")[0]
            if name != "":
                cl.append(name)
            else:
                cl.append("NA")
        else:
            cl.append("NA")

        if "o__" in f:
            name = f.split("o__")[1].split("|f__")[0].replace(".","")
            if "_unclassified" in name:
                name = 'unclassified_' + name.split("_unclassified")[0]
            if name != "":
                order.append(name)
            else:
                order.append("NA")
        else:
            order.append("NA")

        if "f__" in f:
            name = f.split("f__")[1].split("|g__")[0].replace(".","")
            if "_unclassified" in name:
                name = 'unclassified_' + name.split("_unclassified")[0]
            if name != "":
                family.append(name)
            else:
                family.append("NA")
        else:
            family.append("NA")

        if "g__" in f:
            name = f.split("g__")[1].split("|s__")[0].replace(".","")
            if "_unclassified" in name:
                name = 'unclassified_' + name.split("_unclassified")[0]
            if name != "":
                genus.append(name)
            else:
                genus.append("NA")
        else:
            genus.append("NA")

        if "s__" in f:
            name = f.split("s__")[1]
            if "_unclassified" in name:
                name = 'unclassified_' + name.split("_unclassified")[0]
            if name != "":
                species.append(name)
            else:
                species.append("NA")
        else:
            species.append("NA")
    if len(species) == 0:
        d = {'kingdom': kingdom, 'phylum': phylum, 'class':cl,
            'order':order, 'family':family, 'genus':genus}
        feature_df = pd.DataFrame(data=d)
        feature_df.index = feature_df['genus']
    else:
        d = {'kingdom': kingdom, 'phylum': phylum, 'class':cl,
            'order':order, 'family':family, 'genus':genus, 'species': species}
        feature_df = pd.DataFrame(data=d)
        feature_df.index = feature_df['species']
    return feature_df


# In[5]:


# a function to remove repeated data without changing data order
def duplicate_data(list1):
    list2 = []
    for i in list1:
        if i not in list2:
            list2.append(i)
    return list2


# In[6]:


# input features_raw Output feature_df, features and features_level
def generate_features_with_levels(features_raw):
    feature_df = get_feature_df(features_raw)
    features = []
    features_level = []
    for level in feature_df.columns.values:
        for item in feature_df[level].tolist():
            if item != "NA" :
                features.append(item+"|"+str(level))
    features = duplicate_data(features)
    print("There are "+str(len(features))+" validied features")
    for i in range(len(features)):
        features_level.append(features[i].split("|")[1])
        features[i] = features[i].split("|")[0]
    return feature_df,features,features_level


# In[7]:


def generate_map(data,features_raw):
    data = data.transpose()
    feature_df, features, features_level = generate_features_with_levels(features_raw)
    map_values = []
    for j in range(1,len(data.columns)+1):
        value_df = feature_df
        value_df.index= data[j].tolist()
        values = []
        for i in range(len(features)):
            sum_df = value_df[value_df[features_level[i]] == features[i]]
            values.append(sum(sum_df.index.values))
        map_values.append(np.array(values))
    map_values = np.array(map_values)
    levels_num = {"kingdom":0,"phylum":1,"class":2,"order":3,"family":4,"genus":5,"species":6}
    levels = feature_df.columns.values
    map_edges = []
    for i in range(len(features)):
        lines = []
        df_i = feature_df[feature_df[features_level[i]]==features[i]]
        for j in range(len(features)):
            if abs(levels_num[features_level[i]]-levels_num[features_level[j]]) == 1:
                if len(df_i[df_i[features_level[j]]==features[j]]) != 0:
                    lines.append(1)
                else:
                    lines.append(0)
            else:
                lines.append(0)
        map_edges.append(lines)
    map_edges = np.array(map_edges)
    print("shape of map_edges:"+str(map_edges.shape))
    print("shape of map_values:"+str(map_values.shape))
    return features,features_level,map_values, map_edges


# In[59]:


def generate_Network(features,features_level,map_values,map_edges):
    graph_edges = []
    for i in range(len(map_edges)):
        for j in range(len(map_edges)):
            if map_edges[i][j] == 1:
                graph_edges.append((i,j))
    graph = networkx.Graph()
    graph.add_nodes_from(range(len(features)))
    for i in range(len(features)):
        graph.nodes[i]['level'] = features_level[i]
        graph.nodes[i]['name'] = features[i]
    graph.add_edges_from(graph_edges)
    return graph

def prepare_GNN_data(map_values,map_edges):
    node_labels = []
    graph_indicator = []
    for i in range(map_values.shape[0]):
        for x in map_values[i]:
            node_labels.append(x)
            graph_indicator.append(i)
    graph_edges = []
    for i in range(len(map_edges)):
        for j in range(len(map_edges)):
            if map_edges[i][j] == 1:
                graph_edges.append((i,j))
    graph_start = []
    graph_end = []
    for i in range(len(graph_edges)):
        graph_start.append(graph_edges[i][0])
        graph_end.append(graph_edges[i][1])
    edges_list = [[],[]]
    for i in range(map_values.shape[0]):
        for j in range(len(graph_edges)):
            edges_list[0].append(graph_start[j]+i*map_edges.shape[0])
            edges_list[1].append(graph_end[j] + i*map_edges.shape[0])
    return np.array(node_labels), edges_list, np.array(graph_indicator)


# In[52]:


if __name__ == "__main__":
    data,labels = load_data("Cirrhosis")
    data = data.transpose()
    sums = data.sum(axis=1)
    data = data.divide(sums, axis=0)
    labels, label_set = pd.factorize(labels)
    features_raw = list(data.columns.values)
    features,features_level,map_values, map_edges = generate_map(data,features_raw)


# In[ ]:




