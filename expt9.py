import nltk
from nltk.tokenize import word_tokenize
from pprint import pprint
from nltk.chunk import conlltags2tree, tree2conlltags, ne_chunk
nltk.download('maxent_ne_chunker')
nltk.download('words')


text = input("Enter Paragraph: \n")
punc = '''~`!@#$%^&*()-_=+[]{}';:"/.,<>?\|'''

for ele in text:
    if ele in punc:
        text = text.replace(ele,"")

print("\nRemoval of punctuations:\n")
print(text)

text_tokens = word_tokenize(text)

pos = nltk.pos_tag(text_tokens)

pattern = 'NP: {<DT>?<JJ>*<NN>}'
cp = nltk.RegexpParser(pattern)
cs = cp.parse(pos)
print("\n\nChunk:\n")
print(cs)


entities = nltk.chunk.ne_chunk(pos)
entities.draw()
cs.draw()

iob = tree2conlltags(entities)
print("IOB Tagging: \n")
pprint(iob)

print("\n\n NER:n")
print(entities)