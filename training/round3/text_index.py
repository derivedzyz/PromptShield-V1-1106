"""Shared exact and near-duplicate text index for dataset splits."""
from collections import defaultdict
import math
import unicodedata
import zlib


def key(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

class TextIndex:
    """Exact text and >=0.85 Jaccard similarity of normalized 5-character shingles."""
    def __init__(self):
        self.keys = set()
        self.sets = []
        self.prefix = defaultdict(list)
        self.cache = {}

    def describe(self, text):
        normal = key(text)
        if normal not in self.cache:
            simple = "".join(normal.split())
            grams = {zlib.crc32(simple[i:i+5].encode()) for i in range(max(1,len(simple)-4))}
            prefix = sorted(grams)[:len(grams)-math.ceil(.85*len(grams))+1]
            self.cache[normal] = (grams, prefix)
        return normal, *self.cache[normal]

    def contains(self, text):
        normal, grams, prefix = self.describe(text)
        if normal in self.keys:
            return True
        candidates = {idx for h in prefix for idx in self.prefix.get(h,())}
        for idx in candidates:
            other = self.sets[idx]
            if min(len(other),len(grams)) < .85*max(len(other),len(grams)):
                continue
            common = len(other & grams)
            if common >= .85*(len(other)+len(grams)-common):
                return True
        return False

    def add(self, text):
        normal, grams, prefix = self.describe(text)
        if normal in self.keys:
            return
        self.keys.add(normal)
        for h in prefix:
            self.prefix[h].append(len(self.sets))
        self.sets.append(grams)
