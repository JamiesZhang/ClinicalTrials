import requests
from pprint import pprint
import re
from lxml import etree

topic_path = 'topics2017.xml'
ex_topic_path = '../ext_topics.xml'

base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
fetch_url = '{}esearch.fcgi?db={}&term={}'

id_pattern = re.compile(r'<Id>(.*?)</Id>')
entry_pattern = re.compile(r'Entry Terms:(.*?)Previous Indexing:', re.DOTALL)
entry_pattern2 = re.compile(r'Entry Terms:(.*?)All MeSH Categories', re.DOTALL)
head_pattern = re.compile(r'1: ')

def parse_syn(split_text):
    syn = []
    start = False
    for line in split_text:
        if line.strip() == 'syn {':
            start = True
            continue
        if start:
            if line.strip() == '}':
                break
            else:
                syn.append(line.strip(' ').strip('",')) #stupid!
    return syn, start

def parse_entryterms(split_text):
    entry_terms = []
    start = False
    for line in split_text:
        if line.strip() == 'Entry Terms:':
            start = True
            continue
        if start:
            if line.strip() == '':
                break
            else:
                entry_terms.append(line.strip())
    return entry_terms, start

def find_all_relavant_diseases(term):
    id_resp = requests.get(fetch_url.format(base, 'mesh', term))
    id_list = re.findall(id_pattern, id_resp.text)
    all_terms = []
    for id in id_list:
        id_fetch_url = '{}efetch.fcgi?db={}&id={}'.format(base, 'mesh', id)
        resp = requests.get(id_fetch_url)
        split_text = resp.text.split('\n')
        terms, start = parse_entryterms(split_text)
        if start is False:
            print(id_fetch_url, 'No Entry Terms')
        for t in terms:
            all_terms.append(t)
    return all_terms
        
def find_all_relavant_genes(gene):
    id_resp = requests.get(fetch_url.format(base, 'gene', gene))
    id_list = re.findall(id_pattern, id_resp.text)
    all_genes = []
    for id in id_list:
        id_fetch_url = '{}efetch.fcgi?db={}&id={}'.format(base, 'gene', id)
        resp = requests.get(id_fetch_url)
        split_text = resp.text.split('\n')
        genes, start = parse_syn(split_text)
        if start is False:
            print(id_fetch_url, 'No syn')
        for g in genes:
            all_genes.append(g)
    return all_genes

# extract term in topic
topic_xml = etree.parse(topic_path, etree.XMLParser())
diseases = topic_xml.xpath('//disease/text()')
genes = topic_xml.xpath('//gene/text()')
demographic = topic_xml.xpath('//demographic/text()')
other = topic_xml.xpath('//other/text()')

ex_topics = open(ex_topic_path,'a')
for i in range(len(diseases)):
    ex_topics.write("<topic_number=\"{}\">\n".format(i+1))
    relevant_disease = find_all_relavant_diseases(diseases[i])
    ex_topics.write("\t<disease>{}</disease>\n".format(diseases[i] + '|' + ','.join(relevant_disease)))
    relevant_gene = find_all_relavant_genes(genes[i])
    relevant_gene.append(genes[i])
    ex_topics.write("\t<gene>{}</gene>\n".format('|'.join(relevant_gene)))
    ex_topics.write("\t<demographic>{}</demographic>\n".format(demographic[i]))
    ex_topics.write("\t<other>{}</other>\n".format(other[i]))
    ex_topics.write("</topic>\n")
ex_topics.close()