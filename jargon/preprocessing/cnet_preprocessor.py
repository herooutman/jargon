#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""parse_wot.py: Description of what this file does."""
__author__ = "Kan Yuan"
__copyright__ = "Copyright 2016, Cry Little Kan"
__version__ = "0.1"

import json
import traceback
from bs4 import BeautifulSoup

from .preprocessor import Preprocessor
from monster.database import connect_sqlite


class CnetPreprocessor(Preprocessor):
    name = "cnet"

    def _collect_meta_from_db(self, dbfile):
        with connect_sqlite(dbfile) as conn:
            cursor = conn.cursor()
            sql = "select response_body, url from cnet where response_url like 'https://www.cnet.com/forums/discussions/%';"
            try:
                cursor.execute(sql)
                for x in cursor.fetchall():
                    self.tasks.append(x)
            except Exception:
                traceback.print_exc()
        self.all_loaded = True
        self.logger.info("collected {} tasks".format(len(self.tasks)))

    def _create_table(self):
        sql = """CREATE TABLE IF NOT EXISTS cnet (
                url text PRIMARY KEY,
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
            url = content.get("url", None)
            comments = json.dumps(content.get("comments", list()))
            authors = json.dumps(list(content.get("authors", set())))
            raw_comments = json.dumps(content.get("raw_comments", list()))
            if url:
                data = (url, title, comments, raw_comments, authors, texts)
                c = self.db_writer.cursor()
                sql = "INSERT OR REPLACE INTO reddit(url, title, comments, raw_comments, authors, texts) VALUES (?,?,?,?,?,?)"
                c.execute(sql, data)
                self.db_writer.commit()

    @staticmethod
    def _parse_raw(task):
        response_body, url = task
        res = dict(title="", raw_comments=list(), url=url, authors=set())
        soup = BeautifulSoup(response_body, 'lxml')
        for script in soup(["script", "style"]):
            script.extract()
        soup.find("div", {"class": "side"}).extract()  # remove side bar
        title_tag = soup.find("a", {"class": "title"})
        title_text = title_tag.get_text(" ").strip()
        res["title"] = title_text
        usertext_tags = soup.find_all("div", {"class": "usertext-body"})
        # TODO get authors

        for usertext_tag in usertext_tags:
            usertext_tag_text = usertext_tag.get_text(" ").strip()
            if usertext_tag_text == "[deleted]":
                continue
            res["raw_comments"].append(str(usertext_tag))
        return res
