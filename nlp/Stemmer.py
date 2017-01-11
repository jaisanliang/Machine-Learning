"""
Given a modified version of the Harvard Inquirer
Dictionary, creates positive and negative word lists
with common derivations of included words
"""

import csv
import re

# determines the part of speech of a word from the tag
def findPartOfSpeech(speechTag):
    startTag = speechTag[:4]
    if startTag == "SUPV":
        return "verb"
    elif startTag == "Noun":
        return "noun"
    elif startTag == "Modi":
        return "modifier"
    else:
        return "other"

# add suffixes to adverbs and adjectives
def addSuffix(word,partOfSpeech):
    if partOfSpeech == "verb":
        return addSuffixVerb(word)
    elif partOfSpeech == "noun":
        return addSuffixNoun(word)
    elif partOfSpeech == "modifier":
        return addSuffixMod(word)
    elif partOfSpeech == "other":
        return addSuffixOther(word)

# subroutines for adding suffixes to different
# parts of speech
def addSuffixVerb(word):
    derivList = [word]
    # append "-ed"
    if word[-1] == "E":
        derivList.append(word+"D")
    else:
        derivList.append(word+"ED")
    # append "-ing"
    if word[-1] == "E":
        derivList.append(word[:-1]+"ING")
    else:
        derivList.append(word+"ING")
    # append "-s"
    derivList.append(word+"S")
    return derivList

def addSuffixNoun(word):
    derivList = [word]
    # append "-s"
    if word[-1] == "Y":
        derivList.append(word[:-1]+"IES")
    else:
        derivList.append(word+"S")
    return derivList

def addSuffixMod(word):
    derivList = [word]
    # append "-ly"
    if word[-2:] != "LY":
        if word[-2:] == "LE":
            derivList.append(word[:-1]+"Y")
        else:
            derivList.append(word+"LY")
    return derivList

def addSuffixOther(word):
    derivList = [word]
    return derivList

"""
Harvard Dictionary contains the columns: word,
positive, negative, and part of speech

How to create Harvard Modified list

delete all columns except entry, source, positiv, negativ, and othtags
remove all words which are not positiv or negativ (reduces to 4000 words from 12000)
delete source column (only "upheld" is in Lvd but not in Harvard)

tag analysis
http://www.wjh.harvard.edu/~inquirer/kellystone.htm
SUPV = verb
	"ed", "ing", "s"
Noun = noun
	"s", if ends with "y" strip "y" and add "ies"
Modif = adjective or adverb
	add "ly" to end, if ends with "le" strip off "e" and add "y"
LY
	add "ly" to end if not already "ly" at end
Det
Handels
INTJ
PREP = "against", "short", "worth"
Place = only occurance is "dreamland"

python script does the rest of the processing
"""

readList = open("HarvardModified.csv", "r")
readCSV = csv.reader(readList, delimiter = ",")

negList = open("HarvardNeg.csv", "wb")
posList = open("HarvardPos.csv", "wb")
writeNeg = csv.writer(negList, delimiter = ",")
writePos = csv.writer(posList, delimiter = ",")

rowCount = 0
for row in readCSV:
    # print row
    rowCount += 1
    
    word = row[0]
    # strip out numbers
    sliceIndex = word.find("#")
    if sliceIndex != -1:
        word = word[:sliceIndex]
    isPos = row[1] == "Positiv"
    
    speechTag = row[3]
    partOfSpeech = findPartOfSpeech(speechTag)
    if isPos:
        # write all derivations into positive word list
        for derivation in addSuffix(word,partOfSpeech):
            # print derivation
            writePos.writerow([derivation])
    else:
        for derivation in addSuffix(word,partOfSpeech):
            writeNeg.writerow([derivation])

    # for debugging
    # if rowCount > 10:
        # break

readList.close()
negList.close()
posList.close()
