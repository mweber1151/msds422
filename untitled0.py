# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 08:33:49 2019

@author: jjschued
"""
#need to install azure sdk pip install azure-storage-blob
from azure.storage.blob import BlockBlobService
from azure.storage.blob import ContentSettings
from azure.storage.blob import PublicAccess
block_blob_service = BlockBlobService(is_emulated=True)
import  pandas as pd
#in conda command prompt
#export HTTP_PROXY="http://proxy.rockwellcollins.com:9090"
#export HTTPS_PROXY="https://proxy.rockwellcollins.com:9090"
'''
import os
proxy = 'http://proxy.rockwellcollins.com:9090'
proxys = 'https://proxy.rockwellcollins.com:9090'
os.environ['http_proxy'] = proxy
os.environ['https_proxy'] = proxys
os.environ['HTTP_PROXY'] = proxy
os.environ['HTTPS_PROXY'] = proxys
'''

#need to install azure sdk pip install azure-storage-blob
from azure.storage.blob import BlockBlobService
from azure.storage.blob import ContentSettings

#block_blob_service2 = BlockBlobService(account_name='allisonpoc', sas_token='sv=2018-03-28&ss=bfqt&srt=sco&sp=rl&se=2020-06-10T21:29:51Z&st=2019-06-10T13:29:51Z&spr=https&sig=d5eM%2BKxufaQUXCSSHFrjfHoYk0yYyM45yLscL6uaqw8%3D')
#
#sastoken = '?sv=2018-03-28&ss=bfqt&srt=sco&sp=rl&se=2020-06-10T21:29:51Z&st=2019-06-10T13:29:51Z&spr=https&sig=d5eM%2BKxufaQUXCSSHFrjfHoYk0yYyM45yLscL6uaqw8%3D
#
## List the blobs in the container.
#print("\nList blobs in the container")
#generator = block_blob_service2.list_blobs('allisonpoccontainer')
#for blob in generator:
#    print("\t Blob name: " + blob.name)

block_blob_service3 = BlockBlobService(account_name='utasfullflightarchive', sas_token ='sv=2018-03-28&ss=b&srt=sc&sp=rl&se=2019-12-31T14:24:15Z&st=2019-06-06T05:24:15Z&spr=https&sig=hWJ7GFhGsnVUJt2LWvM524GoTfBJKG%2B9abA1UqocfGc%3D')
generator = block_blob_service3.list_blobs('datalake', num_results= 5000, timeout= 240, prefix="CPLData/")
for blob in generator:
    print("\t Blob name: " + blob.name)
    
block_blob_service3 = BlockBlobService(account_name='utasfullflightarchive', sas_token ='sv=2018-03-28&ss=b&srt=sc&sp=rl&se=2019-12-31T14:24:15Z&st=2019-06-06T05:24:15Z&spr=https&sig=hWJ7GFhGsnVUJt2LWvM524GoTfBJKG%2B9abA1UqocfGc%3D')
generator = block_blob_service3.list_blobs('datalake', timeout= 940, prefix="CPLData/")
number = 0
CPLIST = []
for blob in generator:
    print("\t Blob name: " + blob.name)
    number = number + 1
    CPLIST.append(blob.name)


len(CPLIST)