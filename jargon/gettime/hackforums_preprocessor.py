import json

from bs4 import BeautifulSoup
from monster.database import connect_sqlite
from .preprocessor import Preprocessor
from datetime import datetime

class HackforumsPreprocessor(Preprocessor):
    name = "hackforums"

    def _collect_meta_from_db(self, db_file):
        total_tasks = 0
        with connect_sqlite(db_file) as conn:
            cursor = conn.cursor()
            sql = "SELECT tid, html FROM raw_html;"
            cursor.execute(sql)
            for x in cursor.fetchall():
                tid, html = x
                total_tasks += 1
                self.tasks.put(dict(tid=tid, html=html))
        self.all_loaded.value = True
        self.total_tasks = total_tasks
        self.logger.info("collected {} tasks".format(self.total_tasks))

    def _collect_meta_from_fs(self):
        pass

    @staticmethod
    def _parse_raw(task, logger):
        tid = task['tid']
        html = task['html']

        soup = BeautifulSoup(html, 'lxml')
        for script in soup(["script", "style"]):
            script.extract()
        div = soup.find('div', id='content')
        if not div:
            return

        nav = div.find('div', {'class': 'navigation'})
        posts = div.find('div', id='posts')
        if not posts or not nav:
            return

        allurls = nav.find_all('a')
        if len(allurls) < 3:
            return

        title = nav.find('span').get_text(" ")
        raw_comments = list()
        dates = list()

        for post in posts.find_all('', {'class': 'tborder'}):
            tcat = post.find('td', {'class': 'tcat'})
            date_div = tcat.find('div', {'class': 'float_left smalltext'})
            if date_div:
                datestr = date_div.get_text().strip().split("\n")[0].strip()
                try:
                    dt = datetime.strptime(datestr, "%m-%d-%Y, %I:%M %p")
                except ValueError:
                    print("ERROR:", date_div)
                    continue
                for comment in post.find_all('div', {'class': 'post_body'}):
                    raw_comments.append(str(comment))
                    dates.append(dt.timestamp())

        authors = set([
            x.find('span', {'class': 'largetext'}).get_text()
            for x in posts.find_all('td', {'class': 'post_author'})
        ])

        res = dict(
            title=title, raw_comments=raw_comments, tid=tid, authors=authors, dates=dates)
        return res

    def _create_table(self):
        pass
    def _insert_data(self, item):
        pass
