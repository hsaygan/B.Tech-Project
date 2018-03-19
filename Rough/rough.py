def process_random(file_path):
    with open(file_path, buffering=3) as f:
        for line in f:
            print(line)

process_random("textfile.txt")
