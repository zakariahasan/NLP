import pandas as pd
import numpy as np
import plac
import random
from pathlib import Path
import spacy
from tqdm import tqdm 
from spacy.util import minibatch, compounding
from spacy import displacy
import string
from spacy.gold import GoldParse
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')


# ## Data preprocessing




df = pd.read_csv("data.csv")

# Convert json file to spaCy format.
import plac
import logging
import argparse
import sys
import os
import json
import pickle

@plac.annotations(input_file=("Input file", "option", "i", str), output_file=("Output file", "option", "o", str))

def main(input_file=None, output_file=None):
    try:
        training_data = []
        lines=[]
        with open(input_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                point = annotation['points'][0]
                labels = annotation['label']
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    entities.append((point['start'], point['end'] + 1 ,label))


            training_data.append((text, {"entities" : entities}))

        print(training_data)

        with open(output_file, 'wb') as fp:
            pickle.dump(training_data, fp)

    except Exception as e:
        logging.exception("Unable to process " + input_file + "\n" + "error = " + str(e))
        return None
if __name__ == '__main__':
    plac.call(main)# Convert .tsv file to dataturks json format. 
import json
import logging
import sys
def tsv_to_json_format(input_path,output_path,unknown_label):
    try:
        f=open(input_path,'r') # input file
        fp=open(output_path, 'w') # output file
        data_dict={}
        annotations =[]
        label_dict={}
        s=''
        start=0
        for line in f:
            if line[0:len(line)-1]!='.\tO':
                word,entity=line.split('\t')
                s+=word+" "
                entity=entity[:len(entity)-1]
                if entity!=unknown_label:
                    if len(entity) != 1:
                        d={}
                        d['text']=word
                        d['start']=start
                        d['end']=start+len(word)-1  
                        try:
                            label_dict[entity].append(d)
                        except:
                            label_dict[entity]=[]
                            label_dict[entity].append(d) 
                start+=len(word)+1
            else:
                data_dict['content']=s
                s=''
                label_list=[]
                for ents in list(label_dict.keys()):
                    for i in range(len(label_dict[ents])):
                        if(label_dict[ents][i]['text']!=''):
                            l=[ents,label_dict[ents][i]]
                            for j in range(i+1,len(label_dict[ents])): 
                                if(label_dict[ents][i]['text']==label_dict[ents][j]['text']):  
                                    di={}
                                    di['start']=label_dict[ents][j]['start']
                                    di['end']=label_dict[ents][j]['end']
                                    di['text']=label_dict[ents][i]['text']
                                    l.append(di)
                                    label_dict[ents][j]['text']=''
                            label_list.append(l)                          
                            
                for entities in label_list:
                    label={}
                    label['label']=[entities[0]]
                    label['points']=entities[1:]
                    annotations.append(label)
                data_dict['annotation']=annotations
                annotations=[]
                json.dump(data_dict, fp)
                fp.write('\n')
                data_dict={}
                start=0
                label_dict={}
    except Exception as e:
        logging.exception("Unable to process file" + "\n" + "error = " + str(e))
        return None

tsv_to_json_format("zak_ner.tsv",'zak_ner.json','abc')
entity=[]
entity_types=[]
for n in range(0,len(df['meta'])):
    if type(df['meta'].loc[n])==str:
        if 'SIZE' in list(eval(df['meta'].loc[n]).keys()):
            size_list=eval(df['meta'].loc[n])['SIZE']
            for m in range(0,len(size_list)):
                size_list_deep=size_list[m].lstrip().rstrip().upper()
                if size_list_deep not in entity:
                    if size_list_deep not in size_exceptions:
                        entity.append(size_list_deep)
                        entity_types.append('SIZE')
        if 'COLOR' in list(eval(df['meta'].loc[n]).keys()):
            color_list=eval(df['meta'].loc[n])['COLOR']
            for m in range(0, len(color_list)):
                color_item=color_list[m].title()
                if ', ' in color_item:
                    color_item_list=color_item.split(', ')
                    for l in range(0,len(color_item_list)):
                        color_item_list_deep=color_item_list[l].lstrip().rstrip()
                        if len(color_item_list_deep)!=0:
                            if color_item_list_deep not in entity:
                                if color_item_list_deep.isnumeric()==False:
                                    if color_item_list_deep not in color_exceptions:
                                        entity.append(color_item_list_deep)
                                        entity_types.append('COLOR')
                                            
                else:
                    color_item_list_deep=color_list[m].lstrip().rstrip().title()
                    if len(color_item_list_deep)!=0:
                        if color_item_list_deep not in entity:
                            if color_item_list_deep.isnumeric()==False:
                                if color_item_list_deep not in color_exceptions:
                                    entity.append(color_item_list_deep)
                                    entity_types.append('COLOR')   
        entity_2=[]
entity_types_2=[]
titles_unique_brand=[]
description_unique_brand=[]
for n in range(0,len(df['brand'])):
    if df['brand'].loc[n] not in entity_2:
        if df['brand'].loc[n] != '•':
            entity_2.append(df['brand'].loc[n])
            entity_types_2.append('BRAND')
            titles_unique_brand.append(df['title'].loc[n])
            if type(df['description'].loc[n])==str:
                description_unique_brand.append(df['description'].loc[n])entity_3=[]
entity_types_3=[]
for n in range(0,len(df['meta'])):
    if type(df['meta'].loc[n])==str:
        if 'GENDER' in list(eval(df['meta'].loc[n]).keys()):
            gender_list=eval(df['meta'].loc[n])['GENDER']
            for m in range(0, len(gender_list)):
                gender_list_deep=gender_list[m].lstrip().rstrip().title()
                if gender_list_deep not in entity_3:
                    if (gender_list_deep != '[Unisex]' and gender_list_deep != '[Female]' and gender_list_deep != '[Male]'):
                        gender_list_deep_list=gender_list_deep.split(', ')
                        for gender_list_deep_list_item in gender_list_deep_list:
                            if gender_list_deep_list_item not in entity_3:
                                entity_3.append(gender_list_deep_list_item)
                                entity_types_3.append('GENDER')
        if 'AGE_GROUP' in list(eval(df['meta'].loc[n]).keys()):
            age_group_list=eval(df['meta'].loc[n])['AGE_GROUP']
            for m in range(0, len(age_group_list)):
                age_group_list_deep=age_group_list[m].lstrip().rstrip().title()
                if age_group_list_deep not in entity_3:
                    entity_3.append(age_group_list_deep)
                    entity_types_3.append('AGE_GROUP')



def obtain_annotation(title):
    
    # Initial annotation, charactors' ranges are highly overlapped
    entity_set_list=[]
    entity_dict={}
    entity_set_range_list=[]
    for m in range(0,len(entity)):
        if entity[m] in title:
            index_i=title.find(entity[m])
            index_f=index_i+len(entity[m])
            if (index_i != 0) and (index_f != len(title)):
                if (title[index_i-1] == ' ' and title[index_f] == ' '):
                    #print(entity[m])
                    entity_tuple=(index_i, index_f, entity_types[m])
                    entity_set_list.append(entity_tuple)
                    entity_set_range_list.append(range(index_i, index_f))
            if (index_i == 0) and (index_f != len(title)):
                if (title[index_f] == ' '):
                    #print(entity[m])
                    entity_tuple=(index_i, index_f, entity_types[m])
                    entity_set_list.append(entity_tuple)
                    entity_set_range_list.append(range(index_i, index_f))
            if (index_i != 0) and (index_f == len(title)):
                if (title[index_i-1] == ' '):
                    #print(entity[m])
                    entity_tuple=(index_i, index_f, entity_types[m])
                    entity_set_list.append(entity_tuple)
                    entity_set_range_list.append(range(index_i, index_f))

    # Second Step: Get rid of overlapped charactors' ranges
    entity_set_list_2=[]
    for n, entity_set_range_1 in enumerate(entity_set_range_list):
        entity_set_range_test=set(entity_set_range_1)
        inter=0
        entity_set_list_2_temp=[]
        for m, entity_set_range_2 in enumerate(entity_set_range_list):
            if entity_set_range_1 != entity_set_range_2:
                interss=entity_set_range_test.intersection(entity_set_range_2)
                #entity_set_list_2_temp=[]
                if interss==set():
                    inter += 1
                else:
                    if set(entity_set_range_1)>set(entity_set_range_2):
                        if entity_set_list[n] not in entity_set_list_2_temp:
                            entity_set_list_2_temp.append(entity_set_list[n])
                    elif set(entity_set_range_1)<set(entity_set_range_2):
                        if entity_set_list[m] not in entity_set_list_2_temp:
                            entity_set_list_2_temp.append(entity_set_list[m])
        
        if m == inter:
            if entity_set_list[n] not in entity_set_list_2:
                entity_set_list_2.append(entity_set_list[n])
        else:
            for entity_set_list_2_temp_item in entity_set_list_2_temp:
                if entity_set_list_2_temp_item not in entity_set_list_2:
                    entity_set_list_2.append(entity_set_list_2_temp_item)
    
    # Third Step: Get rid of overlapped charactors' ranges further if any
    entity_set_list_3=[]
    for n, item_1 in enumerate(entity_set_list_2):
        item_1_range=range(item_1[0],item_1[1])
        inter=0
        entity_set_list_2_temp=[]
        for m, item_2 in enumerate(entity_set_list_2):
            item_2_range=range(item_2[0],item_2[1])
            if item_1_range != item_2_range:
                interss=set(item_1_range).intersection(item_2_range)
                if interss == set():
                    inter += 1
                else:
                    if set(item_1_range)>set(item_2_range):
                        if item_1 not in entity_set_list_2_temp:
                            entity_set_list_2_temp.append(item_1)
                    elif set(item_1_range)<set(item_2_range):
                        if item_2 not in entity_set_list_2_temp:
                            entity_set_list_2_temp.append(item_2)
        if m == inter:
            entity_set_list_3.append(item_1)
        else:
            for entity_set_list_2_temp_item in entity_set_list_2_temp:
                if entity_set_list_2_temp_item not in entity_set_list_3:
                    entity_set_list_3.append(entity_set_list_2_temp_item)
    
    # Fourth Step: Get rid of overlapped charactors' ranges further if any
    entity_set_list_4=[]
    for n, item_1 in enumerate(entity_set_list_3):
        item_1_range=range(item_1[0],item_1[1])
        inter=0
        entity_set_list_2_temp=[]
        for m, item_2 in enumerate(entity_set_list_3):
            item_2_range=range(item_2[0],item_2[1])
            if item_1_range != item_2_range:
                interss=set(item_1_range).intersection(item_2_range)
                if interss == set():
                    inter += 1
                else:
                    if set(item_1_range)>set(item_2_range):
                        if item_1 not in entity_set_list_2_temp:
                            entity_set_list_2_temp.append(item_1)
                    elif set(item_1_range)<set(item_2_range):
                        if item_2 not in entity_set_list_2_temp:
                            entity_set_list_2_temp.append(item_2)
        if m == inter:
            entity_set_list_4.append(item_1)
        else:
            for entity_set_list_2_temp_item in entity_set_list_2_temp:
                if entity_set_list_2_temp_item not in entity_set_list_4:
                    entity_set_list_4.append(entity_set_list_2_temp_item)
    
    # Construct annotation
    entity_dict['entities']=entity_set_list_4
    annotation_n=(title, entity_dict)
    return annotation_n


# In[ ]:


TRAIN_DATA_0=[]
for n in range(0,len(titles_unique_brand)):
    annotation=obtain_annotation(titles_unique_brand[n])
    if annotation[1]['entities']!=[]:
        TRAIN_DATA_0.append(annotation)


# In[ ]:


TRAIN_DATA_1=[]
for n in range(0,len(description_unique_brand)):
    annotation=obtain_annotation(description_unique_brand[n])
    if annotation[1]['entities']!=[]:
        TRAIN_DATA_1.append(annotation)
TRAIN_DATA = TRAIN_DATA_0 + TRAIN_DATA_1


# In[ ]:





# In[ ]:




with open("Ner_Pickle.txt", "wb") as fp:   #Pickling
    pickle.dump(TRAIN_DATA, fp)
# ## Annotated Data

# In[3]:


with open("Ner_Pickle.txt", "rb") as fp:   
    DATA = pickle.load(fp)
TRAIN_DATA=DATA[0:100]
TEST_DATA=DATA[101:200]


# ## Build Model

# In[4]:


def built_model(data,iterations,drop=0.2):
    TRAIN_DATA = data
    nlp = spacy.blank('xx')  
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes): 
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  
                    [annotations],  
                    drop=drop,  
                    sgd=optimizer,  
                    losses=losses)
            print(losses)
    return nlp


# ## Train Model

# In[5]:


get_ipython().run_cell_magic('time', '', 'nlp = built_model(TRAIN_DATA,10)')


# In[6]:


pickle.dump(nlp, open('model.pkl','wb'))


# ## Evaluate Model

# In[21]:


tp=0
tr=0
tf=0
ta=0
c=0.
# GENERATING THE CLASSIFICATION REPORT
for text,annot in TEST_DATA:
    doc_to_test=nlp(text)

    d={}
    for ent in doc_to_test.ents:
        d[ent.label_]=[]
    for ent in doc_to_test.ents:
        d[ent.label_].append(ent.text)
   
    for ent in doc_to_test.ents:
        d[ent.label_]=[0,0,0,0,0,0]
    for ent in doc_to_test.ents:
        doc_gold_text= nlp.make_doc(text)
        gold = GoldParse(doc_gold_text, entities=annot.get("entities"))
        y_true = [ent.label_ if ent.label_ in x else 'Not '+ent.label_ for x in gold.ner]
        y_pred = [x.ent_type_ if x.ent_type_ ==ent.label_ else 'Not '+ent.label_ for x in doc_to_test]  
        # print(text,y_pred)
        if(d[ent.label_][0]==0):
            #print("For Entity "+ent.label_+"\n")   
            #f.write(classification_report(y_true, y_pred)+"\n")
            (p,r,f,s)= precision_recall_fscore_support(y_true,y_pred,average='weighted')
            a=accuracy_score(y_true,y_pred)
            d[ent.label_][0]=1
            d[ent.label_][1]+=p
            d[ent.label_][2]+=r
            d[ent.label_][3]+=f
            d[ent.label_][4]+=a
            d[ent.label_][5]+=1
    c+=1
#print(d)
for i in d:
    #print("\n For Entity "+i+"\n")
    print("Accuracy : "+str((d[i][4]/d[i][5])*100)+"%")
    print("Precision : "+str(round(d[i][1]/d[i][5],2)))
    print("Recall : "+str(round(d[i][2]/d[i][5],2)))
    print("F-score : "+str(round(d[i][3]/d[i][5],2)))


# ## Save Model

# In[9]:


output_dir=Path("/Users/zakaria/anaconda3/ML-Material/Integrify/project/sp_zak")
if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)
    print("Saved model to", output_dir)#

    # test the saved model
    print("Loading from", output_dir)
    nlp2 = spacy.load(output_dir)
    for text, _ in TEST_DATA:
        doc = nlp2(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        #print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])


# ## Test with Test Data



for text, _ in TEST_DATA[0:3]:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])




for n in range(0,20):
    doc = nlp(TEST_DATA[n][0])
    spacy.displacy.render(doc, style="ent")





