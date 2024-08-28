import requests
from bs4 import BeautifulSoup

response=requests.get('https://namu.wiki/w/런던/%EA%B4%80%EA%B4%91')
soup=BeautifulSoup(response.content,'html.parser')
# #mw-content-text > div.mw-content-ltr.mw-parser-output > p:nth-child(5)
# mw-content-text > div.mw-content-ltr.mw-parser-output > p:nth-child(5)
#app > main > div._2c1r0Xom > div.wedMMTTN > article > div.zPg43W5X > div > div:nth-child(2) > div > div > div:nth-child(7) > div
##app > main > div._2c1r0Xom > div.wedMMTTN > article > div.zPg43W5X > div > div:nth-child(2) > div > div > div:nth-child(7)
selector='#app > main > div._2c1r0Xom > div.wedMMTTN > article > div.zPg43W5X > div > div:nth-child(2) > div > div > div:nth-child(7)'
#app > main > div._2c1r0Xom > div.wedMMTTN > article > div.zPg43W5X > div > div:nth-child(2) > div > div > div:nth-child(7)
elements=soup.select(selector)

for element in elements:
    print(element.text)
