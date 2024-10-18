import json
import csv
import sys
import unicodedata

def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0]!="C")

def process_json_file_to_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

        writer.writerow(['title', 'answer'])

        for topic in data:
            print(topic)
            comments = topic['comments']
            if len(comments) < 2:
                continue
            for comment in comments:
                if comment['author'] == 'u/AutoModerator':
                    continue
    
                writer.writerow([remove_control_characters(topic['title']), remove_control_characters(comment['body'])])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_json_file> <output_csv_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    process_json_file_to_csv(input_file, output_file)