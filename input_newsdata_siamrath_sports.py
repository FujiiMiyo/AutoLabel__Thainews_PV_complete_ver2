# -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:28:48 2020


"""

from urllib.request import urlopen
from bs4 import BeautifulSoup
import datetime
import random
import pymysql
import sys
import re

sys.setrecursionlimit(23000)

conn = pymysql.connect(host = 'localhost', user = 'root', passwd = None, db = 'news', charset = 'utf8')

cur = conn.cursor()
cur.execute("USE news") #***

random.seed(datetime.datetime.now())

cnt = 0
sql = cur.execute("SELECT URL FROM siamrath_url_sports") #***edit categories: economic, education, entertainment, foreign, it, sports
url = cur.fetchall()

def store(URL,website,title,genre,body):
    cur.execute("INSERT INTO article(URL,website,title,genre,body) VALUES(\"%s\",\"%s\",\"%s\",\"%s\",\"%s\")",(URL,website,title,genre,body))
    cur.connection.commit()
    
def has_class_but_no_id(tag):
    return tag.has_attr('class') and not tag.has_attr('id')
    
def getLinks(articleUrl,genre,website):
    global cnt
    cnt +=1
    try:
        articleUrl = articleUrl.replace("\'", "")
        html = urlopen(articleUrl)
        bsObj = BeautifulSoup(html,"lxml")
        
        URL = articleUrl
        
        title = bsObj.find("h1",{"class":"page-header"}).get_text()
        print(title)
        #body = bsObj.find("section",{"class":"article-detail"}).find(has_class_but_no_id).get_text().replace('\n', '').strip()
        #body = bsObj.find("div",{"class":"col-xs-12 col-sm-10 col-sm-offset-1"}).get_text().replace('\n', '').replace("'","").replace("(","").replace(")","").replace("\xa0","").replace("\r\n","").replace("\r","").replace(";","").replace("\t","").replace("\u200b","").replace("/","").replace("-","").replace("googletag.cmd.push","").replace("function","").replace("googletag.display","").replace("div","").replace("gpt","").replace("ad","").replace("8668011","").replace("{","").replace("}","").replace(",","").replace("\'","").replace("\"","").replace("!!","").replace("!","").replace(":","").replace("”","").replace("“","").replace("","").replace("?","").replace("{","").replace("}","").replace("’","").replace("‘","").replace("–","").replace(".","").replace("\[","").replace("\]","").strip()
        body = bsObj.find("div",{"class":"col-xs-12 col-sm-10 col-sm-offset-1"}).get_text().replace('\n', '').replace("'","").replace("(","").replace(")","").replace("\xa0","").replace("\r\n","").replace("\r","").replace(";","").replace("\t","").replace("\u200b","").replace("/","").replace("-","").replace("googletag.cmd.push","").replace("function","").replace("googletag.display","").replace("div","").replace("gpt","").replace("ad","").replace("8668011","").replace("{","").replace("}","").replace(",","").replace("\'","").replace("\"","").replace("!!","").replace("!","").replace(":","").replace("”","").replace("“","").replace("","").replace("?","").replace("{","").replace("}","").replace("’","").replace("‘","").replace("\[","").replace("\]","").replace("*","").strip()
        print(body)        
        '''time = bsObj.find("span",{"class":"mr5"}).get_text()
        print(time)'''
    
        '''siamrath = "สยามรัฐออนไลน์"
        for submitted in bsObj.find_all('span',  attrs={'class': 'submitted'}):
            submitted_descendants = submitted.descendants
            for d in submitted_descendants:
                if d.name == 'span' and d.get('class', '') == ['mr5']:
                    for e in d:
                        if e not in d.name != 'a' and d.text != siamrath:
                           time = e
                           print(time)'''
                    
        
        store(URL,website,title,genre,body)
        print(URL)
    except Exception:
        print(cnt)
        return 0
    
    
if __name__ == '__main__':
    for i in url:
        getLinks(i[0],"sports","siamrath") #***edit categories: economic, education, entertainment, foreign, it, sports
        
    print("finish")