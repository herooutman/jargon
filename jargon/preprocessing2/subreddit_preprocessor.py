#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""parse_wot.py: Description of what this file does."""
__author__ = "Kan Yuan"
__copyright__ = "Copyright 2016, Cry Little Kan"
__version__ = "0.1"

import sys
import os
import json

from bs4 import BeautifulSoup
from monster.database import connect_sqlite
from .preprocessor import Preprocessor


class SubredditPreprocessor(Preprocessor):
    name = "reddit"

    def _collect_meta_from_db(self, db_file):
        with connect_sqlite(db_file) as conn:
            cursor = conn.cursor()
            sql = "SELECT url, response_body FROM subreddit;"
            cursor.execute(sql)
            for x in cursor.fetchall():
                url, html = x
                self.tasks.append((url, html))
        self.logger.info("collected {} tasks".format(len(self.tasks)))

    def _create_table(self):
        sql = """CREATE TABLE IF NOT EXISTS reddit (
                url text PRIMARY KEY,
                title text,
                comments text,
                raw_comments text,
                texts text
                );"""
        # c = conn.cursor()
        self.db_conn.execute(sql)
        self.db_conn.commit()

    def _insert_data(self, item):
        content, texts = item
        if not content:
            pass
        else:
            title = content.get("title", "")
            url = content.get("url", None)
            comments = json.dumps(content.get("comments", list()))
            raw_comments = json.dumps(content.get("raw_comments", list()))
            if url:
                data = (url, title, comments, raw_comments, texts)
                c = self.db_conn.cursor()
                sql = "INSERT OR REPLACE INTO reddit(url, title, comments, raw_comments, texts) VALUES (?,?,?,?,?)"
                c.execute(sql, data)
                self.db_conn.commit()

    def _parse_raw(self, task):
        url, html = task
        res = dict(title="", raw_comments=list(), url=url, path="")
        soup = BeautifulSoup(html, 'lxml')
        for script in soup(["script", "style"]):
            script.extract()
        soup.find("div", {"class": "side"}).extract()  # remove side bar
        title_tag = soup.find("a", {"class": "title"})
        title_text = title_tag.get_text(" ").strip()
        res["title"] = title_text
        usertext_tags = soup.find_all("div", {"class": "usertext-body"})
        for usertext_tag in usertext_tags:
            usertext_tag_text = usertext_tag.get_text(" ").strip()
            if usertext_tag_text == "[deleted]":
                continue
            res["raw_comments"].append(str(usertext_tag))
        return res
