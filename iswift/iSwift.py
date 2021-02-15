
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import yaml
import itertools
import csv
import datetime

#read configuration from config file
config_file = open('config.yaml')
config = yaml.load(config_file,Loader=yaml.FullLoader)

conTHR = config['recomm_threshold']['confidence']
filepath = config['input_path']
dims_len = config['dims_len']


search_step=0
error_item=0
item_cnt=0

def read_file(local_filepath):
    data_dict = {}
    start_dict = {}
    layer_dict = {}

    f = open(local_filepath+"root_cause",'r')
    _list=f.readlines()    
    f.close()
    root_cause_list=[x.strip() for x in _list]
    root_cause = [tuple(x.split(",")) for x in root_cause_list]
    print("root cause",root_cause)

    for i in range (1,dims_len+1):
        filename = local_filepath +str(i)+".csv"
        with open(filename, "r", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            firstline = True
            for line in reader:
                if firstline:
                    firstline = False
                    continue
                line[dims_len]=int(float(line[dims_len]))
                line[dims_len+1]=int(float(line[dims_len+1]))
                key=tuple(line[0:dims_len])
                value = tuple(line[-2:])
                data_dict[key]=value

                layer = dims_len-key.count("*")
                if(layer not in layer_dict.keys()):
                    layer_dict[layer] = {}
                    layer_dict[layer][key] = value
                else:
                    layer_dict[layer][key] = value
                if(dims_len-key.count("*")==1):
                    start_dict[key]=value
                
                if (key in root_cause):
                    
                    latent_force_temp = value[1] / (error_item)
                    confidence_set_temp = value[1] / (value[1] + value[0])
                    support_temp = value[1] / item_cnt
                    print("root-cause info",key, latent_force_temp, confidence_set_temp, support_temp,error_item)
                
    return data_dict,start_dict,layer_dict,root_cause
    
#get the candidates    
def search_Tree(data_dict,start_dict,layer_dict):
    latent_force = {}
    confidence_set = {}
    sp_set = {}
    search_set={}
    recommond_list=[]
    global search_step
    for item in start_dict:  #search the nodes at the 1th layer
        search_step = search_step + 1
        error = start_dict[item][1]
        normal = start_dict[item][0]

        latent_force[item]=error/(error_item)
        confidence_set[item]=error/(error+normal)
        sp_set[item] = latent_force[item]+confidence_set[item]
        if(latent_force[item]<config['cut_threshold']): #prun by latent force
            continue
        search_set[item]=sp_set[item] 

        if(confidence_set[item]> conTHR): 
            conf_avg = subNodeCalc(item,layer_dict,data_dict)
            if (abs(confidence_set[item] - conf_avg) < config["con_combine_thr"]): #select by confidence loss
                recommond_list.append(item)
                removeChildfromList(item,layer_dict,data_dict)
            else:
                continue
   
    search_set_sorted= sorted(search_set.items(), key=lambda item:item[1], reverse=True)
    Candidate_list = getCandidateList(search_set_sorted,search_set)
    search_set.clear()

    for loop in range(2,config['for_num']+2):
            for item in Candidate_list:
                
                if(item not in data_dict.keys()):
                    continue
                abnormal = data_dict[item][1]
                normal = data_dict[item][0]
                search_step = search_step + 1
                latent_force[item]=abnormal/(error_item)
                confidence_set[item]=abnormal/(abnormal+normal)
                sp_set[item] = latent_force[item]+confidence_set[item]

                if(latent_force[item]<config['cut_threshold']):
                    continue
                search_set[item]=sp_set[item]
                
                if(confidence_set[item]> conTHR):
                    conf_avg = subNodeCalc(item,layer_dict,data_dict)
                    if (abs(confidence_set[item] - conf_avg) < config["con_combine_thr"]):
                        recommond_list.append(item)
                        removeChildfromList(item,layer_dict,data_dict)
                    else:
                        continue
            print(loop," search_set：" + str(len(search_set)))
            search_set_sorted= sorted(search_set.items(), key=lambda item:item[1], reverse=True)
            Candidate_list = getCandidateList(search_set_sorted,search_set)
            search_set.clear()
    return latent_force,confidence_set,recommond_list

#filter the candidates by pod score
def make_pod(recommond_list,latent_force):
    pod_dict = {}
    for recom in recommond_list:
        rx = ""
        for i in range(0,len(recom)):
            if(recom[i]=="*"):
                rx = rx+"*"
            else:
                rx = rx + str(i)
            if(i < len(recom)-1):
                rx = rx + ","
        if( rx not in pod_dict.keys() ):
            pod_dict[rx] = []
        pod_dict[rx].append(recom)
    
    pod_filter_dict = {}
    for pod_item in pod_dict:
        pod_dict[pod_item] = quick_sort(pod_dict[pod_item],latent_force)
        pod_filter_dict[pod_item] = 0
        sum_t = 0
        for item in pod_dict[pod_item]:
            sum_t = sum_t + latent_force[item]
        pod_filter_dict[pod_item] = sum_t /len(pod_dict[pod_item])

    pod_filter_sorted = sorted(pod_filter_dict.items(), key=lambda x: x[1], reverse=True)
    j=0
    for item in pod_filter_sorted:
        if j == 3:
            break
        j = j + 1
        print("recommend",j,item[0],item[1],pod_dict[item[0]])
    return pod_dict,pod_filter_sorted

#calculate the performance  
def calc_f1(pod_dict,pod_filter_sorted,root_cause_list,dims_list_num):
    tn_num = 0
    if(len(pod_filter_sorted)==0) :
        return [],[],[],tn_num
    if( len(pod_dict[pod_filter_sorted[0][0]])>0 ):
        rcmd = pod_dict[pod_filter_sorted[0][0]]
        tp = []
        fp = []
        fn = []
        tp = [val for val in root_cause_list if val in rcmd]
        for item in rcmd:
            if (item not in root_cause_list):
                fp.append(item)
        fn = [val for val in root_cause_list if val not in rcmd]
        tn_num = dims_list_num - len(rcmd) - len(fn)

    return tp,fp,fn,tn_num

#search root-cause set M_opt
def fbeem(indexxx,local_filepath):
    global search_step
    starttime = datetime.datetime.now()
    data_dict,start_dict,layer_dict,root_cause = read_file(local_filepath)
    dims_list_num = len(data_dict)
    #get the candidates
    latent_force,confidence_set,recommond_list = search_Tree(data_dict,start_dict,layer_dict)

    endtime = datetime.datetime.now()
    #candidates filtering
    pod_dict,pod_filter_sorted = make_pod(recommond_list,latent_force)
   
    tp,fp,fn,tn_num=calc_f1(pod_dict,pod_filter_sorted,root_cause,dims_list_num)
    print("performance", indexxx, str(len(tp)), str(len(fp)), str(len(fn)), str(tn_num), search_step,(endtime - starttime).seconds)


def quick_sort(nums,latent_force):
    n = len(nums)
    if n ==1 or len(nums)==0:
        return nums  
    left = []
    right = []
    for i in range(1,n):
        if latent_force[nums[i]] >= latent_force[nums[0]]:
            left.append(nums[i])
        else:
            right.append(nums[i])         
    return quick_sort(left,latent_force)+[nums[0]]+quick_sort(right,latent_force)

#prun the subgraph nodes         
def removeChildfromList(_list,layer_dict,data_dict):
    global search_step
    layer = dims_len-_list.count("*")
    for l in range(layer+1,dims_len+1):
        for item in layer_dict[l]:
            mark=0
            for i in range(0,len(_list)):
                if(_list[i]!="*" and _list[i]!=item[i]):
                    mark=1
                    break
            if (mark == 0):
                if (item in data_dict.keys()):
                    data_dict.pop(item)

#get the expand set
def getCandidateList(search_set_sorted,search_set):
    global search_step
    i=0
    result=[]
    topK=[]
    for item in search_set_sorted:
        if(i==(config["topK"])):
            break
        search_set.pop(item[0])
        topK.append(item[0])
        i=i+1

    #TOPK merge
    for i in range(0,len(topK)-1):
        for j in range(i+1,len(topK)):
            _list = []
            for k in range(0,dims_len):
                if(topK[i][k]=="*" and topK[j][k]=="*"):
                    _list.append("*")
                    continue
                elif(topK[i][k]!="*" and topK[j][k]!="*"):
                    if(topK[i][k] == topK[j][k]):
                        _list.append(topK[i][k])
                    else:
                        break
                else:
                    if(topK[i][k]!="*"):
                        _list.append(topK[i][k])
                    else:
                        _list.append(topK[j][k])

            if(len(_list)==dims_len):
                result.append(tuple(_list))
    
    return result

#get the children nodes 
def subNodeCalc(_list, layer_dict, data_dict):
    global search_step
    conf_avg = 0
    result_list = []
    layer = dims_len - _list.count("*")
    if(layer>=5):
        return 0
    search_dict = layer_dict[layer + 1]
    for item in search_dict:
        if (item not in data_dict.keys()):
            continue
        mark = 0
        for i in range(0, len(_list)):
            if (_list[i] != "*" and _list[i] != item[i]):
                mark = 1
                break
        if (mark == 0):
            result_list.append(item)
    local_conf = {}

    for item in result_list:
        search_step = search_step + 1
        if (item not in data_dict.keys()):
            continue
        error = data_dict[item][1]
        normal = data_dict[item][0]
        local_conf[item] = error / (error + normal)
        conf_avg = conf_avg + local_conf[item]
    if len(result_list) == 0:
        return 0
    else:
        return conf_avg / len(result_list)



if __name__ == "__main__":
    for i in range(0, config["input_num"]):
        search_step=0
        error_item=0  #the total number of abnormal leaf nodes of search graph T
        item_cnt=0
        print(i,": ++++++++++++++++++++++")
        local_filepath = filepath + str(i) + "/"
        df_e = pd.read_csv(local_filepath+"merge.csv",encoding="unicode_escape")
        for index,row in df_e.iterrows():
            item_cnt+=1
            if(row['error']==1):
                error_item+=1
        if(error_item==0):
            continue

        fbeem(i,local_filepath)

