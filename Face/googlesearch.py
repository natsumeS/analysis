import requests
import os
from bs4 import BeautifulSoup
import base64
import time
import sys

'''
Google検索で画像をとってきます。最大80まで
'''


class GoogleSearchAgent:
    '''
    initialize
    save_dir:画像の保存ディレクトリのパス
    '''

    def __init__(self, save_dir):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.wbfn_t_list = []

    '''
    download(image_url, base_filename, *, timeout=10)
    指定されたurlから画像をダウンロードする。
    image_url:画像リソースのurlあるいはデータurl
    base_filename:保存された画像の名前
    '''

    def download(self, image_url, base_filename, *, timeout=10, format_ext="jpeg", size=300):
        filename = "{}/{}.jpg".format(self.save_dir, base_filename)
        if image_url[0:4] == 'http':
            res = requests.get(image_url, allow_redirects=False, timeout=timeout)
            assert res.status_code == 200, "HTTP status error"
            with open(filename, "wb") as file:
                file.write(res.content)
        elif image_url[0:4] == 'data':
            dataurl = str.split(image_url, ',')[1].encode('utf-8')
            with open(filename, "wb") as file:
                file.write(base64.decodebytes(dataurl))

    '''
    検索リストの初期化
    '''

    def clear_searchlist(self):
        self.wbfn_t_list = []

    '''
    add_search(word, cat_name):
    検索リストに追加
    word:検索ワード
    cat_name:検索ワードに対応するカテゴリ・クラス名
    '''

    def add_search(self, word, cat_name):
        self.wbfn_t_list.append((word, cat_name))

    '''
    add_search(word, cat_name):
    検索開始
    img_num:ダウンロードする画像数（最大80)
    file_index_start=1:画像の保存先save_dir/cat_name/index.jpgのindexの開始番号
    '''

    def search(self, img_num=80, file_index_start=1):
        google_url = 'https://www.google.co.jp/search?biw=1707&bih=827&tbm=isch&sa=1&q={}'
        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36'
        }
        for t in self.wbfn_t_list:
            resp = requests.get(google_url.format(t[0]), timeout=1, headers=headers)
            soup = BeautifulSoup(resp.text, "html.parser")
            img_list = soup.find_all("img")
            img_url_list = []
            for img in img_list:
                if img.get('data-src') is not None:
                    img_url_list.append(img.get("data-src"))
            if not os.path.exists("{}/{}".format(self.save_dir, t[1])):
                os.makedirs("{}/{}".format(self.save_dir, t[1]))
            counter = file_index_start
            for img_url in img_url_list:
                self.download(img_url, "{}/{}".format(t[1], counter))
                sys.stdout.write('\rdownload:{}/{}'.format(counter, img_num))
                sys.stdout.flush()
                counter += 1
                if counter > img_num:
                    break
                time.sleep(1)
            print("{} is downloaded".format(t[1]))
