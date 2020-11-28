# -*- coding: utf-8 -*-
# @Time    : 2020/11/28 15:29
# @Author  : zhaogang
# -*- coding: utf-8 -*-
# @Time    : 2020/11/27 9:30
# @Author  : zhaogang
import csv
from sklearn.model_selection import train_test_split
csv.field_size_limit(500 * 1024 * 1024)

def gen_train_data(task):
    tmp=''
    if task=='OCEMOTION':
        tmp='like'
    elif task=='OCNLI':
        tmp='0'
    elif task=='TNEWS':
        tmp='115'

    segment_list=[]
    with open('tianchi_original/'+task+'_train.csv', 'r', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for line in reader:
            segment_list.append(line)
    with open('tianchi_datasets2/'+task+'/train.csv', 'w', newline='', encoding='utf8') as f:  # output csv file
        #writer = csv.writer(f)
        for train in segment_list:
            f.write('\t'.join(train)+'\n')


    segment_list=[]
    with open('tianchi_original/'+task+'_a.csv', 'r', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for line in reader:
            segment_list.append(line)
    with open('tianchi_datasets2/'+task+'/dev.csv', 'w', newline='', encoding='utf8') as f:  # output csv file
        #writer = csv.writer(f)
        for test in segment_list:
            #writer.writerow(test)
            f.write('\t'.join(test)+'\t'+tmp+'\n')

tasks=['OCEMOTION','OCNLI','TNEWS']
for task in tasks:
    gen_train_data(task)