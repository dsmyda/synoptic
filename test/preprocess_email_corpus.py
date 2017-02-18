# Author: Daniel Smyda
# Course: Stat Approaches to NLP
# Professor: Benjamin Wellner
# Date: Dec 18, 2016

import xml.etree.ElementTree as ET

def main():
    corpusTree = ET.parse("bc3&framework.1.1/bc3corpus.1.0/corpus.xml")
    annotationTree = ET.parse("bc3&framework.1.1/bc3corpus.1.0/annotation.xml")

    doc_to_sentence = {}
    doc_to_labels = {}

    corpusRoot = corpusTree.getroot()
    for child in corpusRoot:
        process_corpus_child(child.iter(), doc_to_sentence)

    annotationRoot = annotationTree.getroot()
    for child in annotationRoot:
        process_annotation_child(child, doc_to_labels)

    with open("sentence_extraction/positive.txt", 'w') as p, open("sentence_extraction/negative.txt", 'w') as n:
        for thread in doc_to_sentence:
            for annotator in doc_to_labels[thread]:
                for sentence in doc_to_sentence[thread]:
                        if sentence in doc_to_labels[thread][annotator]:
                            p.write(doc_to_sentence[thread][sentence]+"\n")
                        else:
                            n.write(doc_to_sentence[thread][sentence]+"\n")


def process_corpus_child(child_iterable, doc_to_sentence):
    title = ""
    for element in child_iterable:
        if element.tag == "name":
            title = element.text
            doc_to_sentence[title] = {}
        if element.tag == "Sent":
            doc_to_sentence[title][element.attrib["id"]] = element.text

def process_annotation_child(child, doc_to_labels):
    title = child.find("name").text
    doc_to_labels[title] = {}
    annotations = child.findall("annotation")
    for annotation in annotations:
        annotator = annotation.find("desc").text
        doc_to_labels[title][annotator] = []
        sentences = annotation.find("sentences")
        for sent in sentences.iter():
            if (sent.tag == "sentences"):
                continue
            doc_to_labels[title][annotator].append(sent.attrib["id"])


if __name__ == "__main__":
    main()
