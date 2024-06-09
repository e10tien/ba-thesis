# file name: probing.py
# author: M. Elzinga
# date: 04.05.2024
# purpose: main file for the layer-wise probing research for my bachelor thesis

import pandas as pd
from transformers import AutoTokenizer, AutoModel, TFAutoModel, AutoConfig
import torch
from tqdm import tqdm
from sklearn import svm
from sklearn import metrics
import numpy as np
from datasets import load_dataset


def initiate_model(model_path):
    '''takes model path as input, returns tokenizer, model'''
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # use output_hidden_states to configure the model so it will output the layer embeddings
    config = AutoConfig.from_pretrained(model_path, output_hidden_states=True)
    model = AutoModel.from_pretrained(model_path, config=config)
    return tokenizer, model


def get_accuracy(clf, X_train, X_test, y_train, y_test):
    '''takes classifier model, X_train, X_test, y_train, y_test, 
    returns the accuracy score for the data
    '''
    # train the model
    clf.fit(X_train, y_train)

    # predict labels
    y_pred = clf.predict(X_test)

    #return the accuracy score
    return metrics.accuracy_score(y_test, y_pred)


def extract_embeddings(data, model, tokenizer, layer):
    '''from data as df, model, tokenizer, and layer as int;
    returns the cls token embeddings from the given layer
    '''
    embeds = []
    for sent in tqdm(data["Sentence"]):
        tokenized_text = tokenizer(sent, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**tokenized_text)
        embeds.append(outputs.hidden_states[layer][0,0,:])
    return embeds


def extract_layer(tokenizer, model, layer, train_df, test_df, y_train, y_test):
    '''based on tokenizer, model, layer (int), train dataframe, test dataframe, 
    y_train and y_test, returns accuracy of layer
    '''
    # initiate classifier model
    clf = svm.SVC(kernel='linear', verbose=True)
    # create the embeddings
    train_embeds = extract_embeddings(train_df, model, tokenizer, layer)
    dev_embeds = extract_embeddings(test_df, model, tokenizer, layer)
    # return accuracy
    return get_accuracy(clf, train_embeds, dev_embeds, y_train, y_test)


def extract_nr_embeddings(data, model, tokenizer, start_layer, nr):
    '''takes data, model, tokenizer, start_layer (int), nr (int)
    returns embeddings starting at start_layer, for number (nr) of layers
    (e.g. start_layer = 4, nr = 3; returns embeddings of layers 4 through 7)
    '''
    embeds = {}
    for i in range(nr):
      embeds[i+start_layer] = []

    for sent in tqdm(data["Sentence"]):
        tokenized_text = tokenizer(sent, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**tokenized_text)
        for i in range(nr):
          embeds[i+start_layer].append(outputs.hidden_states[i+start_layer][0,0,:])
    print(embeds)
    return embeds


def extract_nr_layers(tokenizer, model, start_layer, train_df, test_df, y_train, y_test, nr):
    '''takes tokenizer, model, start_layer(int), train dataframe, test dataframe, y_train, 
    y_test, nr(int), where start_layer is start, and nr is number of layers to extract,
    prints layer number and its accuracy
    '''
    train_embeds = extract_nr_embeddings(train_df, model, tokenizer, start_layer, nr)
    dev_embeds = extract_nr_embeddings(test_df, model, tokenizer, start_layer, nr)

    for i in range(nr):
        clf = svm.SVC(kernel='linear', verbose=True)
        acc = get_accuracy(clf, train_embeds[i+start_layer], dev_embeds[i+start_layer], y_train, y_test)
        print(i+start_layer, acc)


def main():
    dataset = load_dataset("GroNLP/dutch-cola")

    # create dataframes from dataset
    train_df = dataset['train']
    # test_df = dataset['validation'] #uncomment to use validation set
    test_df = dataset['test'] #comment when using validation set

    # set the correct data for testing
    y_train = train_df['Acceptability']
    y_test = test_df['Acceptability']

    # models used and their path
    bertje = "GroNLP/bert-base-dutch-cased"
    mbert = "google-bert/bert-base-multilingual-cased"
    robbert = "pdelobelle/robbert-v2-dutch-base"
    xlmr = "FacebookAI/xlm-roberta-base"
    xlmr_l = "FacebookAI/xlm-roberta-large"

    tokenizer, model = initiate_model(bertje) #change parameter to the model you want to use
    start_layer = 1 #set to layer to start
    nr = 4  #can range between 1 and 4 (the amount of layers extracted at the same time)
    extract_nr_layers(tokenizer, model, start_layer, train_df, test_df, y_train, y_test, nr)

    # # extract a single layer
    # layer = 12 #which layer to extract
    # acc = extract_layer(bertje, layer, train_df, test_df, y_train, y_test)
    # print(layer, acc)


if __name__ == '__main__':
    main()
