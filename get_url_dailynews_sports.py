# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 16:06:48 2020


"""

from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import datetime
import random
import pymysql

conn = pymysql.connect(host = 'localhost', user = 'root', passwd = None, db = 'news', charset = 'utf8') #***

cur = conn.cursor()
cur.execute("USE news") #***

random.seed(datetime.datetime.now())
count = 0

def store(URL): 
    cur.execute("INSERT INTO dailynews_url_sports(URL) VALUES (\"%s\")", (URL)) #***edit categories: economic, education, entertainment, foreign, it, sports
    cur.connection.commit()
    
def getLinks(genre,cnt):
    url = "https://www.dailynews.co.th/"+genre+"?page="+str(cnt)
    html = urlopen(url)
    bsObj = BeautifulSoup(html)
    
    #article = bsObj.findAll("article", {"class":"content"})
    for link in bsObj.findAll("a", href = re.compile("^(/" + genre + "/)")):
        if 'href' in link.attrs:
            new_links = "https://www.dailynews.co.th" + link.attrs['href']
            try:
                store(new_links)
                print(new_links)
            except Exception:
                continue
    
    '''for link in bsObj.findAll("a", href = re.compile("https://www.dailynews.co.th/"+genre+"/")):
        if 'href' in link.attrs:
            new_links = link.attrs['href']
            try:
                store(new_links)
                print(new_links)
            except Exception:
                continue'''
    
    if cnt < 200:
        cnt+=1
        print(cnt)
        getLinks(genre,cnt)
            
if __name__ == '__main__':
    getLinks('sports',1) #***edit categories: economic, education, entertainment, foreign, it, sports