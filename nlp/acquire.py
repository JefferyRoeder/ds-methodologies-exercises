import pandas as pd
from requests import get
from bs4 import BeautifulSoup
import re
import numpy as np


def get_blog_articles():
    #loop to pull 5 codeup blogs from cirriculum
    urls = ['https://codeup.com/codeups-data-science-career-accelerator-is-here/','https://codeup.com/data-science-myths/','https://codeup.com/data-science-vs-data-analytics-whats-the-difference/','https://codeup.com/10-tips-to-crush-it-at-the-sa-tech-job-fair/','https://codeup.com/competitor-bootcamps-are-closing-is-the-model-in-danger/']
    headers = {'User-Agent': "Codeup Bayes Data Science"}

    df = pd.DataFrame()
    blogs = []
    titles = []
    for url in urls:

        response = get(url,headers=headers)
        soup = BeautifulSoup(response.text)
        #grabs body of article
        body = soup.find('div',class_="mk-single-content").get_text()

        #grabs title of article
        if soup.find('h1',class_="page-title") is not None:
            title = soup.find('h1',class_="page-title").get_text()
        else:
            title = np.nan

        blogs.append(body)
        titles.append(title)
    df['title'] = pd.Series(titles)
    df['article'] = pd.Series(blogs)
    return df


def get_all_blogs():
    url = 'https://codeup.com/blog/'
    headers = {'User-Agent': "Codeup Bayes Data Science"}
    response = get(url,headers=headers)
    soup = BeautifulSoup(response.text)
    blogs = soup.find('div',class_="vc_tta-panel-body")
    links = blogs.find_all(href=True)

    all_links = [link.attrs['href'] for link in links if re.findall(r'\d+\/\d+',link.attrs['href'])== [] and len(link.attrs['href'])>1]
    all_links = list(set(all_links))
    #loop to pull all codeup blogs
    #urls = ['https://codeup.com/codeups-data-science-career-accelerator-is-here/','https://codeup.com/data-science-myths/','https://codeup.com/data-science-vs-data-analytics-whats-the-difference/','https://codeup.com/10-tips-to-crush-it-at-the-sa-tech-job-fair/','https://codeup.com/competitor-bootcamps-are-closing-is-the-model-in-danger/']

    df = pd.DataFrame()
    blogs = []
    titles = []
    for link in all_links:

        response = get(link,headers=headers)
        soup = BeautifulSoup(response.text)
        body = soup.find('div',class_="mk-single-content").get_text()
        if soup.find('h1',class_="page-title") is not None:
            title = soup.find('h1',class_="page-title").get_text()
        else:
            title = np.nan

        blogs.append(body)
        titles.append(title)
    df['title'] = pd.Series(titles)
    df['article'] = pd.Series(blogs)
    return df