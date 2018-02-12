import json

from bs4 import BeautifulSoup
from monster.database import connect_sqlite
from .preprocessor import Preprocessor


class HackforumsPreprocessor(Preprocessor):
    name = "hackforums"

    def _collect_meta_from_db(self, db_file):
        with connect_sqlite(db_file) as conn:
            cursor = conn.cursor()
            sql = "SELECT tid, html FROM raw_html;"
            cursor.execute(sql)
            for x in cursor.fetchall():
                tid, html = x
                self.tasks.append((tid, html))
        self.all_loaded = True
        self.logger.info("collected {} tasks".format(len(self.tasks)))

    @staticmethod
    def _parse_raw(task):
        tid, html = task
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
        authors = set([
            x.find('span', {'class': 'largetext'}).get_text()
            for x in posts.find_all('td', {'class': 'post_author'})
        ])
        raw_comments = [
            str(x) for x in posts.find_all('div', {'class': 'post_body'})
        ]
        res = dict(
            title=title, raw_comments=raw_comments, tid=tid, authors=authors)
        return res

    def _create_table(self):
        sql = """CREATE TABLE IF NOT EXISTS hackforums (
                tid text PRIMARY KEY,
                title text,
                authors text,
                comments text,
                raw_comments text,
                texts text
                );"""
        self.db_writer.execute(sql)
        self.db_writer.commit()

    def _insert_data(self, item):
        content, texts = item
        if not content:
            pass
        else:
            title = content.get("title", "")
            tid = content.get("tid", None)
            comments = json.dumps(content.get("comments", list()))
            authors = json.dumps(list(content.get("authors", set())))
            raw_comments = json.dumps(content.get("raw_comments", list()))
            if tid:
                data = (tid, title, comments, raw_comments, authors, texts)
                c = self.db_writer.cursor()
                sql = "INSERT OR REPLACE INTO hackforums(tid, title, comments, raw_comments, authors, texts) VALUES (?,?,?,?,?,?)"
                c.execute(sql, data)
                self.db_writer.commit()
