import requests

content = requests.get('https://api.sampleapis.com/coffee/hot')
print(content.text)