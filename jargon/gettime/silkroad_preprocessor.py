#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""parse_wot.py: Description of what this file does."""
__author__ = "Kan Yuan"
__copyright__ = "Copyright 2016, Cry Little Kan"
__version__ = "0.1"

import json
import traceback
import re
import os
from datetime import datetime
from bs4 import BeautifulSoup

from .preprocessor import Preprocessor
from monster.database import connect_sqlite


TOPIC_PAT = re.compile(r'^\s*Author\s+Topic:\s*(.+?)\s*\(Read \d+ times\)\s*$')


class SilkroadPreprocessor(Preprocessor):
    name = "silkroad"

    def __init__(self, **base_argvs):
        super(SilkroadPreprocessor, self).__init__(**base_argvs)

    def _collect_meta_from_fs(self):
        total_tasks = 0
        for subfolder in os.listdir(self.in_data):
            subfolder_path = os.path.join(self.in_data, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            for fn in os.listdir(subfolder_path):
                if fn.startswith("index.php") and fn.split("?")[-1].startswith("topic"):
                    path = os.path.join(subfolder_path, fn)
                    task = dict(path=path)
                    self.tasks.put(task)
                    total_tasks += 1
        self.all_loaded.value = True
        self.total_tasks = total_tasks
        self.logger.info("collected {} tasks".format(total_tasks))
        return

    def _collect_meta_from_db(self):
        raise NotImplementedError

    def _create_table(self):
        sql = """CREATE VIRTUAL TABLE silkroad_fts USING fts3 (
                path text PRIMARY KEY,
                url text,
                category text,
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
            self.logger.warn("no content, this should not happen often")
            pass
        else:
            title = content.get("title", "")
            path = content.get("path", "")
            category = content.get("category", "")
            url = content.get("url", None)
            comments = json.dumps(content.get("comments", list()))
            authors = json.dumps(list(content.get("authors", set())))
            raw_comments = json.dumps(content.get("raw_comments", list()))
            if url:
                data = (path, url, category, title, comments, raw_comments, authors, texts)
                c = self.db_writer.cursor()
                sql = "INSERT OR REPLACE INTO silkroad_fts(path, url, category, title, comments, raw_comments, authors, texts) VALUES (?,?,?,?,?,?,?,?)"
                c.execute(sql, data)
                self.db_writer.commit()

    # TODO !!!
    @staticmethod
    def _parse_raw(task, logger):
        path = task["path"]

        res = dict(title="", raw_comments=list(), path=path, authors=set(), dates=list())
        with open(path) as fp:
            soup = BeautifulSoup(fp, "lxml")
        for script in soup(["script", "style"]):
            script.extract()
        # soup.find("div", {"class": "side"}).extract()  # remove side bar
        content_section = soup.find("div", {'id': 'main_content_section'})

        nav = content_section.find("div", {'class': 'navigate_section'})
        category = json.dumps([x.strip() for x in nav.get_text(" ").split("»")][:-1])
        res['category'] = category

        forumposts = content_section.find("div", {'id': 'forumposts'})
        if not forumposts:
            return res

        raw_title = forumposts.find("div", {'class': 'cat_bar'}).get_text().strip()
        title = re.match(TOPIC_PAT, raw_title).group(1)
        res['title'] = title

        form = forumposts.find("form", {'id': 'quickModForm'})
        res['url'] = form['action']

        for post_div in form.find_all('div', recursive=False):
            author = post_div.find('div', {'class': 'poster'}).find('h4').get_text().strip()
            res['authors'].add(author)
            meta_tag = post_div.find('div', {'class': "flow_hidden"})
            date_tag = meta_tag = meta_tag.find('div', {'class': "smalltext"})
            datestr = date_tag.get_text("\n")
            datestr = datestr.split("\n")[-1][:-1].strip()
            dt = datetime.strptime(datestr, "%B %d, %Y, %I:%M %p")

            comment_tag = post_div.find('div', {'class': "post"})
            res["raw_comments"].append(str(comment_tag))
            res['dates'].append(dt.timestamp())
        return res
