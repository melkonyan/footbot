from lxml import html
import requests
import argparse
import logging
import json

ARCHIVE_URL = 'http://football.ua/newsarc/page{}.html'
COMMENTS_URL = 'http://services.football.ua/api/Comment/Comments?itemId={article_id}&commentType=1&pageIndex=0&pageSize={max_comments}&sort=2&anchor='
logging.basicConfig(level=logging.INFO)


def load_archive_page(archive_page):
    url = ARCHIVE_URL.format(archive_page)
    try:
        page = requests.get(url)
        if page.status_code != 200:
            logging.warning('Status code {} when loading page {}'.format(page.status_code, url))
            return []
        tree = html.fromstring(page.content)
        articles = tree.xpath('//ul[@class="archive-list"]/*/a[1]/@href')
        logging.info('Found {} articles on the page {}.'.format(len(articles), url))
        return articles
    except ConnectionError as e:
        logging.warning('Error occurred while loading {}: {}'.format(url, e))
        return []


def load_comments(article_url, include_subdomains, max_comments=50):
    comments_response = None
    try:
        article_url = article_url + '#page1'
        page = requests.get(article_url, allow_redirects=include_subdomains)
        if page.status_code == 302:
            logging.info('Ignoring redirect url {}'.format(article_url))
            return []
        if page.status_code != 200:
            logging.warning('Status code {} when loading page {}'.format(page.status_code, article_url))
            return []
        comment_load_fun = 'ModComment.Init('
        page_id_start = page.text.find(comment_load_fun)
        if page_id_start < 0:
            logging.error('Couldnt find page_id for page {}'.format(article_url))
            return []
        page_id_start += len(comment_load_fun)
        page_id_end = page.text.find(',', page_id_start, page_id_start+200)
        page_id = page.text[page_id_start: page_id_end]
        comments_url = COMMENTS_URL.format(article_id=page_id, max_comments=max_comments)
        comments_response = requests.get(comments_url)
        if comments_response.status_code != 200:
            logging.warning('Coundt load comments from {}'.format(comments_url))
            return []
        page_json = json.loads(comments_response.text)
        if 'PageComments' not in page_json.keys():
            return []
        comments_json = page_json['PageComments']
        return comments_json or []
    except ConnectionError as e:
        logging.warning('Error occurred while loading {}:{}'.format(article_url, e))
        return []
    except json.decoder.JSONDecodeError:
        logging.warning('Couldnt parse json response from url {}'.format(comments_response))
        return []


class CommentsSaver:

    _file_prefix = 'footballua_comments'
    _file_type = 'json'
    _comments = []

    def __init__(self, backup_every=None, filename=None):
        self._backup_every = backup_every
        if filename:
            self._file_prefix = filename

    def add(self, comments):
        if self._backup_every:
            new_backup_num = (len(self._comments) + len(comments)) // self._backup_every
            old_backup_num = len(self._comments) // self._backup_every
            if  new_backup_num != old_backup_num and old_backup_num > 0:
                self._store('{}_{}.{}'.format(self._file_prefix, old_backup_num, self._file_type))
        self._comments += comments

    def store(self):
        self._store('{}_all.{}'.format(self._file_prefix, self._file_type))

    def _store(self, filename):
        logging.info('Storing {} comments to file {}'.format(len(self._comments), filename))
        with open(filename, mode='w', encoding='utf8') as f:
            json.dump(self._comments, f, ensure_ascii=False)

def main(comments_to_load, include_subdomains=False, out_file=None, backup_every=None):
    if include_subdomains:
        logging.error('We can not parse subdomains yet. Run with --include_subdomains=False')
        return
    logging.info('Will load {} comments, include_subdomains={}'.format(comments_to_load, include_subdomains))
    comments_loaded = 0
    archive_page = 1
    saver = CommentsSaver(filename=out_file, backup_every=backup_every)
    while True:
        article_urls = load_archive_page(archive_page)
        for article_url in article_urls:
            comments = load_comments(article_url, include_subdomains)
            saver.add(comments)
            comments_loaded += len(comments)
            if comments_loaded >= comments_to_load:
                saver.store()
                return
        archive_page += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A crawler for downloading comments and articles from fooball.ua')
    parser.add_argument('num_comments_to_load', type=int, help='Specify how many comments to load.')
    parser.add_argument('--include_subdomains', type=bool, default=False, help='If set, we will download comments from articles from all subdomains, otherwise only from those that have url starting with "football.ua"')
    parser.add_argument('--filename', type=str, default=None, help='Name of the file to which data will be stored')
    parser.add_argument('--backup_every', type=int, default=None, help='After the specified number of entries is loaded, they will be backuped.')
    args = parser.parse_args()
    main(args.num_comments_to_load, args.include_subdomains, args.filename, args.backup_every)