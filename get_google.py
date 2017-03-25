from lxml import html
import requests
from newsplease import NewsPlease as newp
query = 'IBM'
response = requests.get('http://news.google.com/news?q='+query)
if (response.status_code == 200):
    pagehtml = html.fromstring(response.text)
    news = pagehtml.xpath('//h2[@class="esc-lead-article-title"]/a/@href')
articles = newp.download_articles(news)
print(len(articles))
"""for url in news:
    try:
        article = newp.download_article(url)
    except:
        print(article['title'])"""
