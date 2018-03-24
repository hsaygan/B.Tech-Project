import csv

output_file = "new_lexicon.pickle"

def process_random(file_path, starting_line, ending_line, output_path):
    data = 0
    with open(file_path, 'r') as f:
        entire_file = list(csv.reader(f))
        for line in entire_file[starting_line:ending_line]:
            line = list(line)
            print (line)


process_random("textfile.csv", 0, 2)
