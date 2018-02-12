import json

from monster.database import connect_mysql
from .preprocessor import Preprocessor


class NulledPreprocessor(Preprocessor):
    name = "nulled"

    def _collect_meta_from_db(self, _):
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
                tid, author_name, post, seq = x
                post = "" if post is None else post
                post = post.strip()

                if tid not in threads:
                    threads[tid] = dict(id=tid, comments=list(), authors=set())
                threads[tid]["comments"].append((seq, post))
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
            self.tasks.put(t)
            self.total_tasks += 1
        self.all_loaded = True
        self.logger.info("collected {} tasks".format(self.total_tasks))

    def _collect_meta_from_db_batch(self, _):
        dbargs = dict(
            user='kan',
            password='password',
            host='127.0.0.1',
            database='nulled')
        with connect_mysql(dbargs) as conn:
            chunk_size = 1000
            forum_thread = None
            threads = list()
            cursor = conn.cursor()
            sql = "select topic_id, author_name, post, post_date, title from posts INNER JOIN topics ON tid = topic_id order by tid;"
            cursor.execute(sql)
            batch = cursor.fetchmany(chunk_size)
            while batch:
                for x in batch:
                    tid, author_name, post, seq, title = x
                    post = "" if post is None else post
                    post = post.strip()

                    if forum_thread is None:
                        forum_thread = dict(
                            id=tid,
                            title=title,
                            comments=list(),
                            authors=set())
                    elif forum_thread['id'] == tid:
                        pass
                    else:
                        # self.tasks.put(forum_thread)
                        threads.append(forum_thread)
                        self.total_tasks += 1
                        forum_thread = dict(
                            id=tid,
                            title=title,
                            comments=list(),
                            authors=set())
                    forum_thread["comments"].append((seq, post))
                    if author_name:
                        forum_thread["authors"].add(author_name.lower())
                    pass  # end for
                pass  # end while
            # self.tasks.put(forum_thread)
            threads.append(forum_thread)
            self.total_tasks += 1
            print("enqueue")
            for t in threads:
                self.tasks.put(t)
            pass  # end with

        self.all_loaded = True
        self.logger.info("collected {} tasks".format(self.total_tasks))
        pass  # end _collect_meta_from_db

    @staticmethod
    def _parse_raw(task):
        title = task.get("title", "")
        posts = task.get("comments", list())
        authors = task.get("authors", set())
        posts.sort()
        posts = [c[1] for c in posts]
        raw_comments = list()
        for post in posts:
            raw_comments.append(post)
        tid = task.get("id", -1)
        res = dict(
            title=title, raw_comments=raw_comments, tid=tid, authors=authors)
        return res

    def _create_table(self):
        sql = """CREATE TABLE IF NOT EXISTS nulled (
                tid text PRIMARY KEY,
                title text,
                comments text,
                authors text,
                raw_comments text,
                texts text
                );"""
        # c = conn.cursor()
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
                sql = "INSERT OR REPLACE INTO nulled(tid, title, comments, raw_comments, authors, texts) VALUES (?,?,?,?,?,?)"
                c.execute(sql, data)
                self.db_writer.commit()
