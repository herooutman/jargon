import json

from monster.database import connect_mysql
from .preprocessor import Preprocessor


class DarkodePreprocessor(Preprocessor):
    name = "darkode"

    def _collect_meta_from_db(self, _):
        with connect_mysql() as conn:
            threads = dict()

            cursor = conn.cursor()
            sql = "select threadid, posttext, username, seq  from post left outer join user on user.userid=post.userid;"
            cursor.execute(sql)
            for x in cursor.fetchall():
                pid, text, author, seq = x
                text = text.strip()

                if pid not in threads:
                    threads[pid] = dict(id=pid, comments=list(), authors=set())
                threads[pid]["comments"].append((seq, text))
                if author:
                    threads[pid]["authors"].add(author.lower())

                # res = dict(title="", comments=list(), url=url, path=response_file_path)

            sql = "select threadid, threadtitle from thread;"
            cursor.execute(sql)
            for x in cursor.fetchall():
                pid, title = x
                if pid in threads:
                    threads[pid]["title"] = title
                else:
                    threads[pid] = dict(id=pid, title=title)

            self.tasks = threads.values()
        self.all_loaded = True
        self.logger.info("collected {} tasks".format(len(self.tasks)))

    @staticmethod
    def _parse_raw(task):
        title = task.get("title", "")
        raw_comments = task.get("comments", list())
        authors = task.get("authors", set())
        raw_comments.sort()
        raw_comments = [c[1] for c in raw_comments]
        tid = task.get("id", -1)
        res = dict(
            title=title, raw_comments=raw_comments, tid=tid, authors=authors)
        return res

    def _create_table(self):
        sql = """CREATE TABLE IF NOT EXISTS darkode (
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
                sql = "INSERT OR REPLACE INTO darkode(tid, title, comments, raw_comments, authors, texts) VALUES (?,?,?,?,?,?)"
                c.execute(sql, data)
                self.db_writer.commit()