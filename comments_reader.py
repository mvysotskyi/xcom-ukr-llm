import json
import random

class CommentsReader:
    def __init__(self, filename):
        with open(filename) as f:
            self.topics = json.load(f)

        self.topics = self._get_titles_comments()

    def _get_titles_comments(self):
        res = {}

        for topic in self.topics:
            if isinstance(topic['comments'][0], str):
                continue

            title = topic['title']
            res[title] = [comment['body'] for comment in topic['comments']]

        return res
    
    def nrand_formatted(self, n):
        res = ""
        n = min(n, len(self.topics))

        chosen_topics_keys = random.sample(self.topics.keys(), n)

        for key in chosen_topics_keys:
            res += f"TITLE: {key}\nCOMMENTS:"
            res += "\n<|comment_separator|>\n".join(self.topics[key])
            res += "\n\n\n"

        return res


if __name__ == "__main__":
    cr = CommentsReader("data/ukraine_comments.json")
    print(cr.nrand_formatted(3))