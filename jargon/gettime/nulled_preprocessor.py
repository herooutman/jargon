import json

from monster.database import connect_mysql
from .preprocessor import Preprocessor


class NulledPreprocessor(Preprocessor):
    name = "nulled"

    def _collect_meta_from_db(self, _):
        total_tasks = 0
        dbargs = dict(
            user='kan',
            password='password',
            host='127.0.0.1',
            database='nulled')
        with connect_mysql(dbargs) as conn:
            threads = dict()

            cursor = conn.cursor()
            sql = "select topic_id, author_name, post, post_date from posts;"
            cursor.execute(sql)
            for x in cursor.fetchall():
                tid, author_name, post, post_date = x
                post = "" if post is None else post
                post = post.strip()

                if tid not in threads:
                    threads[tid] = dict(id=tid, comments=list(), authors=set())
                threads[tid]["comments"].append((post_date, post))
                if author_name:
                    threads[tid]["authors"].add(author_name.lower())

            sql = "select tid, title from topics;"
            cursor.execute(sql)
            for x in cursor.fetchall():
                tid, title = x
                if tid in threads:
                    threads[tid]["title"] = title
                else:
                    threads[tid] = dict(id=tid, title=title)

        for t in threads.values():
            if "comments" in t:
                self.tasks.put(t)
                total_tasks += 1
        self.total_tasks = total_tasks
        self.all_loaded.value = True
        self.logger.info("collected {} tasks".format(self.total_tasks))


    @staticmethod
    def _parse_raw(task, logger):
        title = task.get("title", "")
        posts = task.get("comments", list())
        authors = task.get("authors", set())
        posts.sort()
        dates = [c[0] for c in posts]
        posts = [c[1] for c in posts]
        raw_comments = list()
        for post in posts:
            raw_comments.append(post)
        tid = task.get("id", -1)
        res = dict(
            title=title, raw_comments=raw_comments, tid=tid, authors=authors, dates=dates)
        return res

    def _collect_meta_from_fs(self):
        pass

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
                sql = "INSERT OR REPLACE INTO nulled(tid, title, comments, raw_comments, authors, texts) VALUES (?,?,?,?,?,?)"
                c.execute(sql, data)
                self.db_writer.commit()
