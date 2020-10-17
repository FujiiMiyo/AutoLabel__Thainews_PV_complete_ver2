# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:23:23 2020


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
    cur.execute("INSERT INTO siamrath_url_foreign_test(URL) VALUES (\"%s\")", (URL)) #***edit categories: economic, education, entertainment, foreign, it, sports
    cur.connection.commit()
    
def getLinks(genre,cnt):
    url = "https://siamrath.co.th/"+genre+"?page="+str(cnt)
    html = urlopen(url)
    bsObj = BeautifulSoup(html)
    
    #article = bsObj.findAll("article", {"class":"content"})
    for link in bsObj.find("div",{"class":"view-content"}).findAll("a",href = re.compile("^(/n/)")):
        if 'href' in link.attrs:
            new_links = "https://siamrath.co.th"+link.attrs['href']
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
    
    if cnt < 50: #*** 0 to 19 (last page = 8 articles) = 20 pages -> Date: 20200401
        cnt+=1
        print(cnt)
        getLinks(genre,cnt)
            
if __name__ == '__main__':
    getLinks('world',0) #***edit categories: sports [0 to 18] = 19 pages -> Date: 20200330   
    
    #***edit categories: economy, education, entertainment, world, it(technology/space/automobile), sports
    # 1 page = 24 articles
    
    #NEW VER. -> categories: economic (111 pages), education (29 pages), entertainment (33 pages), foreign (21 pages), it (23 pages), sports (19 pages) -> Date: 20200330
    #OLD VER. -> categories (76 pages/genre): education[0 to 75], entertainment[0 to 75], it(technology[0 to 53 =54]+automobile[0 to 16 =17]+space[0 to 4  =5]), sports[0 to 75]
