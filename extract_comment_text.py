import json
import argparse


def extract_comment_text(comments_json, comments_text):
    comment_strings = set()
    with open(comments_json, 'r', encoding='utf8') as f:
        comments = json.load(f)
        for comment in comments:
            text = comment['Text']
            text.replace('\n', ' ')
            comment_strings.add(text)
    with open(comments_text, 'w', encoding='utf8') as f:
        for name in comment_strings:
            f.write('{}\n'.format(name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('comments_json', type=str, help='A json file containing comments.')
    parser.add_argument('--comments_text', type=str, default='comments.txt', help='A text file with all comments. Each comment at a new line.')
    args = parser.parse_args()
    extract_comment_text(args.comments_json, args.comments_text)

