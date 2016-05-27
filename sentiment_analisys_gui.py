#!/usr/bin/env python
# -*- coding: utf-8

import Tkinter as tk
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
import codecs, treetaggerwrapper, glob, os, csv

# Create instance
win = tk.Tk()

# Add a title
win.title("GUI")

# Disable resizing the GUI
#win.resizable(0,0)
#
## Modify adding a Label
aLabel = tk.Label(win, text="\nTransform a directory \n"
                            "full of opinions to a single\n"
                            ".csv file")
aLabel.grid(column=1, row=1)
################################

aLabel2 = tk.Label(win, text="\nCreate a POS-tagged \n"
                             "directory from a directory full \n"
                             "of opinions")
aLabel2.grid(column=1, row=4)
################################
aLabel3 = tk.Label(win, text="\n POS-tag a directory full \n"
                             "of documents into a .csv file")
aLabel3.grid(column=1, row=8)

###############################

aLabel4 = tk.Label(win, text="\n Classify \n")
aLabel4.grid(column=1, row=12)


#Position of the labels


#==================
# Functionality (all the functions go here)
#==================

#This functions convert a directory full of documents to a single .csv file
def retrive(directory_path):
    for filename in sorted(glob.glob(os.path.join(directory_path, '*.txt'))):
        with open(filename, 'r') as f:
            important_stuff = f.read().splitlines()
            oneline = [''.join(important_stuff)]
            yield filename.split('/')[-1] + ', ' +str(oneline).strip('[]"')

def munge(directory,directory2):
            test = tuple(retrive(directory))
            with codecs.open(directory2,'w', encoding='utf8') as out:
                csv_out=csv.writer(out, delimiter='|')
                csv_out.writerow(['id','content'])
                for row in test:
                    csv_out.writerow(row.split(', ', 1))

# funcion de taggeo:

def postag_directory(input_directory, output_directory):
    """
    This function POS-tag a directory full of documents (opinions, tweets, comments)


    Args:
        input_directory: The

        output_directory:

    """
    tagger = treetaggerwrapper.TreeTagger(TAGLANG='en')
    # Loop over the files
    all_tags = []
    for filename in sorted(glob.glob(os.path.join(input_directory, '*.txt'))):
        with codecs.open(filename, encoding='utf-8') as f:
            # Read the file
            content = f.read()
            # Tag it
            tags = tagger.tag_text(content)
            # add those tags to the master tag list
            all_tags.append(tags)

    for i , a_list in enumerate(all_tags):
        new_dir_path = output_directory
        path = os.path.join(new_dir_path, "list%d.txt" % i)
        with open(path, "w") as f:
            for item in a_list:
                f.write(item+"\n")
#pruebas de la funcion postag directory:
#input_d = '/Users/user/Desktop/data/pos'
#out_d = '/Users/user/Desktop/data/tagged_pos'
#postag_directory(input_d, out_d)

#Add a postagged column of the corpus
#id|content|POS-tagged_content

def postag_pandas(input_file, output_file):

    def postag_string(s):
        '''Returns tagged text from string s'''
        if isinstance(s, basestring):
            s = s.decode('UTF-8')
        return tagger.tag_text(s)

    # Reading in the file
    all_lines = []
    with open(input_file) as f:
        for line in f:
            all_lines.append(line.strip().split('|', 1))

    df = pd.DataFrame(all_lines[1:], columns = all_lines[0])

    tagger = treetaggerwrapper.TreeTagger(TAGLANG='en')

    df['POS-tagged_content'] = df['content'].apply(postag_string)

# Format fix:
    def fix_format(x):
        '''x - a list or an array'''
        # With encoding:
        out = list(tuple(i.encode().split('\t')) for i in x)
        # or without:
        # out = list(tuple(i.split('\t')) for i in x)
        return out
    df['POS-tagged_content'] = df['POS-tagged_content'].apply(fix_format)
    df['content'] = df['content'].map(lambda x: x.lstrip('"""'''))

    print list(df.columns.values)
    return df.to_csv(output_file, sep = '|', index=False)


#Classification:
def perform_classification(corpus, labels):
    df_content = pd.read_csv(corpus,sep='|').dropna()
    df_labels = pd.read_csv(labels,sep='|').dropna()
    #Vectorizer:
    count_vect = CountVectorizer(ngram_range=(4,4))
    #vectorizamos el texto
    X = count_vect.fit_transform(df_content['content'].values)
    y = df_labels['labels'].values

    #revisamos el corpus
    print '\n corpus \n',X.toarray()
    #revisamos las labels
    print '\n etiquetas \n',y
    #revisamos la dimension del numero de columnas
    num_columnas_del_df = df_labels.shape[0]
    print '\n count col\n',num_columnas_del_df

    #Hacemos validacion cruzada
    kf = KFold(n=num_columnas_del_df, n_folds=10, shuffle=True, random_state=False)
    print '\n\n........Cross validating........\n\n'
    for train_index, test_index in kf:
        print "\nEntrenó con las opiniones que tienen como indice:\n", train_index, \
        "\nProbó con las opiniones que tiene como indice:\n", test_index
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    y_=[ ]
    prediction_=[ ]




    clf = SVC(kernel='linear', C=1,  class_weight='balanced').fit(X_train, y_train)
    #clf = LinearRegression().fit(X_train, y_train)

    #Cross validation metrics, checar esto
    acc_scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=10)
    # f1_scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=10,average='f1_weighted')
    # recall_scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=10,average='recall')

    print '\nacc_scores como arreglo:\n',acc_scores
    #print '\nacc_scores como arreglo:\n',f1_scores
    # print '\nacc_scores como arreglo:\n',recall_scores

    print("\nAccuracy: %0.2f (+/- %0.2f)" % (acc_scores.mean(), acc_scores.std() * 2))

    prediction = clf.predict(X_test)
    y_.extend(y_test)
    prediction_.extend(prediction)
    # (y_true, y_pred, normalize=True
    # print 'prediccion (y_pred):\n',prediction
    # print 'prediccion_:\n',prediction_
    # print 'y_: (y_true)\n',y_


    #Esto se refiere a las clases:
    target_names = [set(y)]
    print '\ncategorias o etiquetas que contiene el dataset:',target_names

    # Calculando desempeño
    #Hay que ver lo del average
    print 'Accuracy              :', accuracy_score(y_, prediction_)
    print 'Precision             :', precision_score(y_, prediction_,average='weighted')
    print 'Recall                :', recall_score(y_, prediction_,average='weighted')
    print 'F-score               :', f1_score(y_, prediction_,average='weighted')
    print '\nClasification report:\n', classification_report(y_,prediction_)
    print '\nConfusion matrix   :\n',confusion_matrix(y_, prediction_)



#==================
# The UI goes here:
#==================


# Changing our Label
#First Option
tk.Label(win, text="Path of the input file:\t").grid(column=2, row=0)
#Second Option
tk.Label(win, text="Path of the input directory:\t").grid(column=2, row=3)
#Third Option
tk.Label(win, text="Path of the input .csv file:\t").grid(column=2, row=6)
#fourth Option
tk.Label(win, text="Path of the training data:\t").grid(column=2, row=10)

# Adding a Textbox Entry widget (path of the input file)
input_directory1 = tk.StringVar()
input_directory2 = tk.StringVar()
input_directory3 = tk.StringVar()


#First optioon
nameEntered1 = tk.Entry(win, width=12, textvariable=input_directory1)
nameEntered1.grid(column=2, row=1)
#Second option

nameEntered2 = tk.Entry(win, width=12, textvariable=input_directory2)
nameEntered2.grid(column=2, row=4)

nameEntered3 = tk.Entry(win, width=12, textvariable=input_directory3)
nameEntered3.grid(column=2, row=8)



# Changing our Label
#First option
tk.Label(win, text="Path of the output file:").grid(column=11, row=0)
#Second option
tk.Label(win, text="Path of the output directory:").grid(column=11, row=3)
#Third option
tk.Label(win, text="Path of the output .csv file:").grid(column=11, row=6)
#Fourth option
tk.Label(win, text="Path of the labels file:").grid(column=11, row=10)




# Adding a Textbox Entry widget (path of the output file)


output_directory1 = tk.StringVar()
output_directory2 = tk.StringVar()
output_directory3 = tk.StringVar()



nameEntered1 = tk.Entry(win, width=12, textvariable=output_directory1)
nameEntered1.grid(column=11, row=1)

nameEntered2 = tk.Entry(win, width=12, textvariable=output_directory2)
nameEntered2.grid(column=11, row=4)

nameEntered3 = tk.Entry(win, width=12, textvariable=output_directory3)
nameEntered3.grid(column=11, row=8)

corpus = tk.StringVar()
labels = tk.StringVar()

nameEntered4 = tk.Entry(win, width=12, textvariable=corpus)
nameEntered4.grid(column=2, row=12)

nameEntered5 = tk.Entry(win, width=12, textvariable=labels)
nameEntered5.grid(column=11, row=12)
#nameEntered4.grid(column=11, row=10)

# Adding a Button
#call the munge function
action = tk.Button(win, text="To .csv",
                   command=lambda:postag_directory(
                       input_directory1.get(),output_directory1.get()))
action.grid(column=20, row=1)




#call the postag function



action2 = tk.Button(win, text="POS-tag documents",
                   command=lambda:postag_directory(
                       input_directory2.get(),output_directory2.get()))
action2.grid(column=20, row=4)



#call the postag_pandas function
action3 = tk.Button(win, text="add a postagged column",
                     command=lambda:postag_pandas(input_directory3.get(), output_directory3.get()))

action3.grid(column=20, row=8)


action4 = tk.Button(win, text="classify",
                    command=lambda:perform_classification(corpus.get(), labels.get()))

action4.grid(column=20, row=12)





#======================
# Start GUI
#======================
win.mainloop()


#Este ya jala sin pedos
#Probar con