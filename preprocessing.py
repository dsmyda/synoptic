# Author: Daniel Smyda
# Course: Stat Approaches to NLP
# Professor: Benjamin Wellner
# Date: Dec 11, 2016

#Need to do a bit of preprocessing to restructure the corpus. This does not yet accomplish what I had in mind.

from pysimstr import SimStr
import os

PATH_SEP = os.path.sep
FOLDER_PATH = "Portuguese_Corpus"+PATH_SEP
DOCUMENTS = FOLDER_PATH + "docs" + PATH_SEP
SUMMARIES = FOLDER_PATH + "summaries" + PATH_SEP

def main():

    #DATABASE CREATION AND DOC LOOKUP TABLE
    simDB = SimStr(idx_size=1, cutoff=.50)
    sentenceToDoc = {}

    #DOCUMENT SENTENCES ADDED TO DB
    index_documents(simDB, sentenceToDoc)

    #MATCH SUMMARY SENTENCES TO DOCUMENT SENTENCES
    match_summaries(simDB, sentenceToDoc)

def index_documents(simDB, sentenceToDoc):
    docs = os.listdir(DOCUMENTS)
    for doc in docs:
        if os.path.isdir(DOCUMENTS+doc):
            for article in os.listdir(DOCUMENTS+doc):
                if isSentenceFile(article):
                    addToDB(DOCUMENTS+doc+PATH_SEP+article, simDB, sentenceToDoc)
        elif isSentenceFile(doc):
            addToDB(DOCUMENTS+doc, simDB, sentenceToDoc)

def isSentenceFile(str):
    return str.endswith(".sents")

def addToDB(fp, simDB, sentenceToDoc):
    article = os.path.basename(fp)
    doc = os.path.basename(os.path.dirname(fp))
    with open(fp, 'r') as f:
        sentences = f.readlines()
        simDB.insert(sentences)
        for sentence in sentences:
            sentenceToDoc[sentence] = (doc, article)

def match_summaries(simDB, sentenceToDoc):
    summary_dir = os.listdir(SUMMARIES)
    for sum in summary_dir:
        with open(SUMMARIES+sum) as f:
            for sentence in f.readlines():
                #TODO: Fix so that the retrieve score works properly
                print(sentence)
                print(simDB.retrieve_with_score(sentence))
                print("--------------------------")

if __name__ == "__main__":
    main()