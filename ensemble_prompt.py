import sklearn.metrics as metrics
import json
import math
def get_map(output):
    output=output.lower()
    sentiment=""
    if "positive" in output or output=='b':
        sentiment="positive"
    elif "neutral" in output or output=='c':
        sentiment="neutral"
    elif "negative" in output or output=='a':
        sentiment="negative"
    return sentiment
#import sklearn.metrics as metrics
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def has_duplicates(lst):
    return len(lst) != len(set(lst))

def get_prob(prob):
    prob=prob.replace("[","")
    prob=prob.replace("]","")
    prob=prob.split(" ")
    while '' in prob:
        prob.remove('')
    new_prob=[]
    for i in prob:
        new_prob.append(float(i))
    return new_prob

def prob_fusion(p1,p2,a):
        prob=[]
        for u in range(len(p2)):
            prob.append(p1[u]+(p2[u]-p1[u])*a)
        return prob
def get_score(p):
    return 2*max(p)+min(p)-1    
#
datasets=["mvsa-s","mvsa-m","masad","t2015","t2017"]

for dataset in datasets:
    path_base="/home/jncsnlp/lxf/answer/"+dataset+"/llava7b-zs-prompt"
    
    an1=[]
    an2=[]
    result=[]
    labels=[]

    ans_file1 = [json.loads(q) for q in open(path_base+".jsonl", "r")]
    ans_file1= get_chunk(ans_file1, 1, 0)
    ans_file2 = [json.loads(q) for q in open(path_base+"-caption"+".jsonl", "r")]
    ans_file2= get_chunk(ans_file2, 1, 0)

    ids=[]
    prob1=[]
    prob2=[]

    prompt1=[]
    prompt2=[]

    for an in ans_file1:
        #print(an["answer"])
        an1.append(get_map(an["answer"]))
        labels.append(an["label"])
        ids.append(an["id"])
        probability=an["probability"]
        prob1.append(get_prob(probability))
        prompt1.append(an["prompt"])
    for an in ans_file2:
        an2.append(get_map(an["answer"]))
        prob2.append(get_prob(an["probability"]))
        prompt2.append(an["prompt"])
    
    label_map=["positive","negative","neutral"]
    p=n=u=o=0
    #threshold change from 0.1 to 0.9
    threshold=0.3

    aa=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for a in aa:
        result=[]
        for i in range(len(labels)):
            #score=max(prob1[i])
            score=2*max(prob1[i])+min(prob1[i])-1
            if score>threshold:
                result.append(an1[i])
            else:
                #平均
                fusion_all=[]
                for l in range(len(prob1[i])):
                    fusion_all.append(prob1[i][l]*a+prob2[i][l]*(1-a))

                r=label_map[fusion_all.index(max(fusion_all))]
                
                result.append(r)
                
            if "positive" in result[-1]:
                p=p+1
            elif "neutral" in result[-1]:
                n=n+1
            elif "negative" in result[-1]:
                u=u+1
            else:
                o=o+1
                
        acc = metrics.accuracy_score(labels, result)
        f1_macro = metrics.f1_score(labels, result, average='macro')
        f1_w=metrics.f1_score(labels, result, average='weighted')
       
        print(acc*100,f1_macro*100,f1_w*100)