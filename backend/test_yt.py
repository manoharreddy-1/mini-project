import urllib.request
import urllib.parse
import re

def search_youtube(query):
    search_url = 'https://www.youtube.com/results?search_query=' + urllib.parse.quote(query)
    req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        html = urllib.request.urlopen(req).read().decode('utf-8')
        video_ids = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', html)
        seen = set()
        unique_ids = [x for x in video_ids if not (x in seen or seen.add(x))][:5]
        print(unique_ids)
    except Exception as e:
        print(e)

search_youtube('python tutorial')
