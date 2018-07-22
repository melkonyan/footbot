import json
import argparse


def extract_user_names(comments_file, names_file):
    user_names = set()
    with open(comments_file, 'r', encoding='utf8') as f:
        comments = json.load(f)
        for comment in comments:
            user_names.add(comment['UserName'])
    with open(names_file, 'w', encoding='utf8') as f:
        for name in user_names:
            f.write('{}\n'.format(name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('comments_file', type=str, help='A json file containing comments.')
    parser.add_argument('--usernames_file', type=str, default='usernames.txt', help='A text file with all user names found in comments.')
    args = parser.parse_args()
    extract_user_names(args.comments_file, args.usernames_file)


